"""Frames in, utterances out. `INTERFACES.md` §2.

**The closing rule (D12):** close when silence exceeds `silence_threshold_ms`,
**or** when the utterance reaches `max_utterance_ms` — whichever fires first.
The values are derived from the latency budget, not guessed; read D25 before
touching `max_utterance_ms`, which is the dominant latency knob and not RTF.

Everything here is integer arithmetic over a frame index. A frame is 32 ms, so
frame `k` covers `[k*32, (k+1)*32)` ms since session start and no floating-point
time ever enters the state machine. The VAD is injected as a `SpeechDetector`,
which is what lets the whole of this file be tested with scripted probabilities
instead of by hoping a neural network fires on a synthetic tone (D56).

Three behaviours worth knowing from outside:

* **A `max_length` close reopens immediately and contiguously** — the next
  utterance starts on the very next frame with `preceding_silence_ms = 0` and no
  pre-roll. Level 5's fragment carry-over depends on that audio being unbroken.
* **A sub-`min_utterance_ms` utterance is dropped, not emitted** (D30), and it
  does not consume an id. The silence gap accumulates across the drop, so
  `preceding_silence_ms == start_ms - previous_emitted.end_ms` always holds.
* **`speaking` is a side channel, not a field on `Utterance`** (D11). It is the
  debounced VAD state, so the indicator goes dark ~160 ms after speech stops
  rather than waiting the 600 ms it takes to close the utterance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Literal

from .. import config as cfg
from .vad import SileroVad, SpeechDetector

#: 16 000 samples/s x 2 bytes = 32 bytes of PCM per millisecond. Used to assert
#: the invariant `len(pcm) == duration_ms * 32` rather than to convert casually.
BYTES_PER_MS = cfg.SAMPLE_RATE * cfg.SAMPLE_WIDTH_BYTES // 1000

#: "no speech has been seen yet" — far enough back that no debounce or silence
#: window can reach it.
_NEVER = -1_000_000


@dataclass(frozen=True)
class Utterance:
    """One closed span of speech. Exactly the fields `INTERFACES.md` §2 lists."""

    id: str
    pcm: bytes                                  # 16 kHz mono int16
    start_ms: int                               # since session start
    end_ms: int
    closed_by: Literal["silence", "max_length"]
    preceding_silence_ms: int                   # gap before this utterance

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def __repr__(self) -> str:  # pcm is up to 128 KB; never print it
        return (
            f"Utterance({self.id}, {self.start_ms}-{self.end_ms} ms, "
            f"{self.duration_ms} ms, {self.closed_by}, "
            f"gap={self.preceding_silence_ms} ms, {len(self.pcm)} B)"
        )


class Segmenter:
    """The VAD state machine. Push frames; take utterances."""

    def __init__(
        self,
        config: cfg.Config | None = None,
        vad: SpeechDetector | None = None,
        on_speaking: Callable[[bool], None] | None = None,
    ) -> None:
        self.config = config or cfg.load_config()
        self.vad = vad if vad is not None else SileroVad()
        self.on_speaking = on_speaking

        c = self.config
        self._pre_roll_frames = c.utterance_pre_roll_ms // cfg.FRAME_MS
        self._tail_frames = c.utterance_tail_pad_ms // cfg.FRAME_MS
        self._silence_frames = _ceil_div(c.silence_threshold_ms, cfg.FRAME_MS)
        self._debounce_frames = _ceil_div(c.vad_speaking_off_debounce_ms, cfg.FRAME_MS)
        self._effective_max_ms = c.max_utterance_ms

        # Holds the current frame plus enough history to back-date an opening
        # utterance by `utterance_pre_roll_ms`.
        self._ring: deque[bytes] = deque(maxlen=self._pre_roll_frames + 1)

        self._index = 0                     # frames pushed so far
        self._vad_on = False                # hysteresis state
        self._speaking = False              # debounced, published
        #: Last frame the VAD called speech. `_NEVER`, not -1: at -1 the very
        #: first frame of a session sits inside the off-debounce window and the
        #: speaking indicator lights up during the opening silence.
        self._last_speech_frame = _NEVER
        self._open = False
        self._start_frame = 0
        #: First frame of *actual speech* in the open utterance, which is not
        #: `_start_frame` — that one is back-dated by the pre-roll. The
        #: `min_utterance_ms` discard measures between these two, never the
        #: padded duration; see `_close`.
        self._first_speech_frame: int | None = None
        self._buf: list[bytes] = []
        self._next_id = 1
        self._last_emitted_end_ms = 0

        self.frames_pushed = 0
        self.speech_frames = 0
        self.utterances_emitted = 0
        self.utterances_discarded = 0

    # -- the runtime knob (D28) -------------------------------------------

    @property
    def effective_max_utterance_ms(self) -> int:
        """What the cap actually is right now; published in `health` (D28)."""
        return self._effective_max_ms

    def set_max_utterance_ms(self, ms: int) -> int:
        """Override the cap at runtime, clamped; returns the value in force.

        Level 6's backpressure controller shrinks this under load and grows it
        back on recovery (D28). The seam exists now, ahead of its caller, because
        retrofitting a live knob into a state machine is how you get one that is
        only read at construction. It is re-read every frame, so a shrink applies
        to the utterance already in flight rather than the one after it.
        """
        self._effective_max_ms = max(
            self.config.max_utterance_floor_ms,
            min(int(ms), self.config.max_utterance_ms),
        )
        return self._effective_max_ms

    # -- the side channel (D11) -------------------------------------------

    @property
    def speaking(self) -> bool:
        return self._speaking

    def _set_speaking(self, value: bool) -> None:
        if value == self._speaking:
            return
        self._speaking = value
        if self.on_speaking is not None:
            # Edge-triggered on purpose. This drives a UI dot over a WebSocket;
            # 31 messages a second is not a side channel, it is a flood.
            self.on_speaking(value)

    # -- the state machine -------------------------------------------------

    def push(self, frame: bytes) -> Utterance | None:
        """Feed one 1024-byte frame. Returns an utterance if this frame closed one."""
        if len(frame) != cfg.FRAME_BYTES:
            raise ValueError(
                f"expected a {cfg.FRAME_BYTES}-byte frame, got {len(frame)}. "
                f"Sources yield exactly {cfg.FRAME_SAMPLES} samples and the size "
                f"is never varied (D23)."
            )

        k = self._index
        self._index += 1
        self.frames_pushed += 1
        self._ring.append(frame)

        probability = self.vad.probability(frame)
        threshold = (
            self.config.vad_release_threshold
            if self._vad_on
            else self.config.vad_speech_threshold
        )
        is_speech = probability >= threshold
        self._vad_on = is_speech
        if is_speech:
            self.speech_frames += 1
            self._last_speech_frame = k
            if self._open and self._first_speech_frame is None:
                self._first_speech_frame = k

        # Published state: on immediately, off after the debounce. Independent of
        # the utterance, which needs 600 ms of silence to be sure and the dot
        # does not.
        self._set_speaking(
            is_speech or (k - self._last_speech_frame) < self._debounce_frames
        )

        if not self._open:
            if is_speech:
                self._open_at(k)
            return None

        self._buf.append(frame)

        # Silence first: when both rules fire on the same frame, the silence
        # close is the better boundary and its end is earlier anyway.
        if (k - self._last_speech_frame) >= self._silence_frames:
            return self._close(self._trimmed_end(k), "silence")

        # Floor to a whole frame: "no utterance exceeds max_utterance_ms" has to
        # hold for caps that are not a multiple of 32 ms too. At the 2000 ms
        # backpressure floor, closing on the first frame *past* the cap would emit
        # 2016 ms and break the guarantee by a frame.
        if (k + 1 - self._start_frame) >= self._effective_max_ms // cfg.FRAME_MS:
            # No trim: this is a cut through the middle of speech, so there is no
            # trailing silence to remove and nothing to gain by looking.
            return self._close(k + 1, "max_length", reopen_at=k + 1)

        return None

    def flush(self) -> Utterance | None:
        """End of stream. Closes an open utterance; `closed_by` reads "silence".

        `INTERFACES.md` §2 admits two values and end-of-stream is much nearer a
        silence close than a length close — it is trimmed the same way. Level 5's
        separate flush rule, for held sentence fragments, is a different thing
        (D29).
        """
        self._set_speaking(False)
        if not self._open:
            return None
        return self._close(self._trimmed_end(self._index - 1), "silence")

    def utterances(self, frames: Iterable[bytes]) -> Iterator[Utterance]:
        """Drive the whole of a source and yield what closes, flushing at the end."""
        for frame in frames:
            utterance = self.push(frame)
            if utterance is not None:
                yield utterance
        tail = self.flush()
        if tail is not None:
            yield tail

    def reset(self) -> None:
        """New session: clear the VAD's carried state, the buffers and the clock."""
        self.vad.reset()
        self._ring.clear()
        self._index = 0
        self._vad_on = False
        self._speaking = False
        self._last_speech_frame = _NEVER
        self._open = False
        self._start_frame = 0
        self._first_speech_frame = None
        self._buf = []
        self._next_id = 1
        self._last_emitted_end_ms = 0
        self._effective_max_ms = self.config.max_utterance_ms
        self.frames_pushed = 0
        self.speech_frames = 0
        self.utterances_emitted = 0
        self.utterances_discarded = 0

    # -- internals ---------------------------------------------------------

    def _open_at(self, k: int) -> None:
        """Open an utterance on frame `k`, back-dated by the pre-roll."""
        history = list(self._ring)          # ends with frame k
        self._start_frame = k - (len(history) - 1)
        self._first_speech_frame = k        # the pre-roll frames are not speech
        self._buf = history
        self._open = True

    def _trimmed_end(self, k: int) -> int:
        """Exclusive end frame: keep the tail pad, drop the rest of the silence.

        Clamped to the start because a `max_length` close can reopen an utterance
        whose last speech frame is already behind it; that produces a zero-length
        span, which the `min_utterance_ms` discard then drops. The audio is not
        lost — the `max_length` utterance that preceded it kept every frame.
        """
        end = min(self._last_speech_frame + 1 + self._tail_frames, k + 1)
        return max(self._start_frame, end)

    def _close(
        self,
        end_frame: int,
        closed_by: Literal["silence", "max_length"],
        reopen_at: int | None = None,
    ) -> Utterance | None:
        start_ms = self._start_frame * cfg.FRAME_MS
        end_ms = end_frame * cfg.FRAME_MS
        pcm = b"".join(self._buf[: end_frame - self._start_frame])

        # How much of this span the VAD actually called speech. Measured between
        # the first and last speech frames, NOT `end_ms - start_ms`: the pre-roll
        # and the tail pad add 320 ms of deliberate padding (D55), which is more
        # than `min_utterance_ms`, so testing the padded duration would let every
        # blip through and quietly cancel D30's guard.
        speech_ms = (
            0
            if self._first_speech_frame is None
            else (self._last_speech_frame - self._first_speech_frame + 1) * cfg.FRAME_MS
        )

        if reopen_at is None:
            self._open = False
            self._buf = []
            self._first_speech_frame = None
        else:
            # Contiguous continuation: no pre-roll, no gap, next frame onward.
            # Speech is re-detected from scratch, so the continuation is subject
            # to the same discard rule as any other utterance.
            self._start_frame = reopen_at
            self._buf = []
            self._first_speech_frame = None

        if speech_ms < self.config.min_utterance_ms:
            # A VAD blip, not speech (D30). Dropped, not emitted, and it does not
            # consume an id — but the silence it sat in keeps accumulating, so
            # the next utterance's `preceding_silence_ms` spans it.
            #
            # Note for Level 4: this is strictly stronger than the transcriber's
            # own `min_utterance_ms` check in INTERFACES.md §3, which sees only
            # `end_ms - start_ms`. Every utterance published from here contains at
            # least `min_utterance_ms` of speech, not merely of audio.
            self.utterances_discarded += 1
            return None

        utterance = Utterance(
            id=f"u_{self._next_id:04d}",
            pcm=pcm,
            start_ms=start_ms,
            end_ms=end_ms,
            closed_by=closed_by,
            preceding_silence_ms=max(0, start_ms - self._last_emitted_end_ms),
        )
        self._next_id += 1
        self._last_emitted_end_ms = end_ms
        self.utterances_emitted += 1
        return utterance


def _ceil_div(numerator: int, denominator: int) -> int:
    """Round up, so a threshold in ms is never satisfied by slightly less time."""
    return -(-numerator // denominator)
