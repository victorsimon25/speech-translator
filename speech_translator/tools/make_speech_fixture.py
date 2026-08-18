"""``python -m speech_translator.tools.make_speech_fixture`` — builds
``assets/speech_fixture.wav``.

**Run once, on Linux; the WAV is the deliverable.** This is a generator, not a
test dependency: nothing in the test suite imports it, and it will not run on
the demo machine. It is committed because an asset whose provenance is "someone
made it once" is worse than one you can rebuild (D56).

**Why the repo needs speech at all.** Silero is a trained model. It rates
silence, a 440 Hz tone, a four-note chord and white noise below 0.05 — all of
which numpy can produce — and it rates real speech at 1.0. Nothing synthesisable
from numpy lands in between reliably: formant-synthesised vowels with a
syllable-rate envelope peak around 0.8 on 6% of frames, which is not something
to assert on. So every *negative* VAD test can be self-contained and no
*positive* one can, and without a positive one the whole stage can be silently
dead — which is exactly what D53 found had happened.

**Why espeak-ng rather than a downloaded clip.** It is formant synthesis: the
output contains no recorded audio and embeds no third-party sample, so vendoring
it raises no licensing question. It is also reproducible from this file. The
default voice is used deliberately — mbrola voices *are* recorded diphones and
would reintroduce the problem this avoids.

**What it is not.** TTS speech has no room tone, no music bed, no overlapping
speakers and no reverb — the things D23 chose Silero to survive. This fixture
proves the wiring and the state machine; it does not stand in for the captured
meeting WAV that `PLAN.md` asks for, and `STATE.md` keeps that owed.
"""

from __future__ import annotations

import ctypes
import sys
import wave
from pathlib import Path

import numpy as np

from .. import config as cfg
from ..logging_setup import force_utf8

#: espeak-ng's synchronous-with-callback mode: it hands us samples instead of
#: opening an audio device, which is the only reason this works headless.
_AUDIO_OUTPUT_SYNCHRONOUS = 2
_espeakCHARS_UTF8 = 0x1000

#: Three phrases with known silences between them, so the fixture has boundaries
#: whose positions are *arithmetic* rather than a matter of opinion. The third is
#: deliberately long enough to be cut by `max_utterance_ms` (4000), so the
#: max-length path is exercised by the fixture and not only by a fake VAD.
SCRIPT: list[tuple[str, int]] = [
    ("", 500),
    ("Hello everyone, can you all hear me okay?", 900),
    ("Yes.", 900),
    (
        "Great. Let's begin with a quick status update, and then I will share "
        "my screen so that we can walk through the results together in detail.",
        1000,
    ),
]


class _Espeak:
    """One initialised espeak-ng, spoken to repeatedly.

    Initialising per phrase truncates the later ones — the first build of this
    fixture came out 9.1 s where the same script had measured 13.9 s, with the
    long third phrase cut in half. `espeak_Initialize` is a once-per-process
    call, so it is made once per process.
    """

    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL("libespeak-ng.so.1")
        except OSError as exc:  # pragma: no cover - platform-dependent
            raise SystemExit(
                f"libespeak-ng.so.1 not loadable ({exc}).\n"
                f"This generator only runs where espeak-ng is installed; the "
                f"committed {cfg.SPEECH_FIXTURE_PATH.name} is what tests use."
            ) from exc

        self._collected: list[np.ndarray] = []

        # (samples, count, events) -> 0 to keep synthesising. The event struct is
        # never read, so it stays an opaque pointer rather than a mirrored layout.
        callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
        )

        def on_samples(samples, count, _events) -> int:
            if count > 0 and samples:
                self._collected.append(np.ctypeslib.as_array(samples, (count,)).copy())
            return 0

        # Held on the instance: a CFUNCTYPE object that gets garbage-collected
        # while C still holds the pointer is a segfault, not an error.
        self._callback = callback_type(on_samples)

        self.rate = int(self.lib.espeak_Initialize(_AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0))
        if self.rate <= 0:  # pragma: no cover - platform-dependent
            raise SystemExit(f"espeak_Initialize failed (returned {self.rate})")
        self.lib.espeak_SetSynthCallback(self._callback)

    def say(self, text: str) -> np.ndarray:
        """Speak `text`, returning int16 samples at `self.rate`."""
        self._collected.clear()
        raw = text.encode("utf-8")
        self.lib.espeak_Synth(raw, len(raw) + 1, 0, 0, 0, _espeakCHARS_UTF8, None, None)
        self.lib.espeak_Synchronize()
        if not self._collected:  # pragma: no cover - platform-dependent
            raise SystemExit(f"espeak-ng produced no audio for {text!r}")
        return np.concatenate(self._collected).astype(np.int16)


def build() -> np.ndarray:
    """The whole fixture as 16 kHz mono int16, ready to write."""
    import soxr  # the same resampler the capture path uses

    engine = _Espeak()
    pieces: list[np.ndarray] = []
    for text, gap_ms in SCRIPT:
        if text:
            speech = engine.say(text)
            # espeak-ng speaks at 22 050 Hz. One-shot `resample` is correct here
            # and would not be in `FrameFormatter`: this is a whole signal, not a
            # chunk of a stream, so there is no delay line to carry (D46).
            if engine.rate != cfg.SAMPLE_RATE:
                speech = soxr.resample(
                    speech.astype(np.float32), engine.rate, cfg.SAMPLE_RATE
                ).astype(np.int16)
            pieces.append(speech)
        pieces.append(np.zeros(cfg.SAMPLE_RATE * gap_ms // 1000, dtype=np.int16))
    return np.concatenate(pieces)


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    out = Path(argv[0]) if argv else cfg.SPEECH_FIXTURE_PATH
    pcm = build()

    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(cfg.CHANNELS)
        wav.setsampwidth(cfg.SAMPLE_WIDTH_BYTES)
        wav.setframerate(cfg.SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    print(
        f"wrote {out}\n"
        f"  {len(pcm) / cfg.SAMPLE_RATE:.2f} s · {cfg.SAMPLE_RATE} Hz mono int16 · "
        f"{out.stat().st_size / 1024:.0f} KB · peak {np.abs(pcm).max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
