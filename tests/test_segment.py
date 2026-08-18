"""Level 2 acceptance, as tests.

Split deliberately in two, because a neural network is not a fixture.

* **The state machine** is driven by a `ScriptedVad` — a list of probabilities.
  Every boundary, discard and gap assertion lives there, exact and deterministic.
  Hoping a synthetic tone makes Silero fire would test the wrong thing anyway.
* **The Silero wrapper** is driven by the real ONNX. Its negative controls are
  synthesisable (silence, a tone, a chord, noise) and its positive control is
  `assets/speech_fixture.wav`, which exists because nothing numpy can generate
  reliably crosses the threshold (D56).

The tests this file exists for are `test_context_is_carried_between_frames` and
`test_lstm_state_is_carried_between_frames`, each with a control proving the
assertion can see the bug it guards. D53 is a bug that shipped in three places
and passed a preflight check, so it gets the same treatment D46 gave the
resampler seam one layer down.

What is *not* here: "on a captured meeting WAV, boundaries land at real pauses
on inspection". That WAV does not exist yet — see `PLAN.md` and `STATE.md`.
"""

from __future__ import annotations

import dataclasses
import json
import time
import wave

import numpy as np
import pytest

from speech_translator import config as cfg
from speech_translator.audio import WavFileSource
from speech_translator.segment import (
    BYTES_PER_MS,
    Segmenter,
    SileroVad,
    SpeechDetector,
    Utterance,
)

# Derived once from the config so a retune shows up as a failing test rather
# than as tests that quietly assert the old numbers.
C = cfg.Config()
FRAME_MS = cfg.FRAME_MS
PRE_ROLL_FRAMES = C.utterance_pre_roll_ms // FRAME_MS          # 4
TAIL_FRAMES = C.utterance_tail_pad_ms // FRAME_MS              # 6
SILENCE_FRAMES = -(-C.silence_threshold_ms // FRAME_MS)        # 19
MAX_FRAMES = C.max_utterance_ms // FRAME_MS                    # 125


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class ScriptedVad:
    """A `SpeechDetector` that reads its answers off a list.

    This is the point of injecting the VAD: the segmenter is arithmetic, and
    arithmetic deserves exact inputs.
    """

    def __init__(self, probabilities) -> None:
        self.probabilities = list(probabilities)
        self.calls = 0
        self.resets = 0

    def probability(self, frame: bytes) -> float:
        p = self.probabilities[self.calls] if self.calls < len(self.probabilities) else 0.0
        self.calls += 1
        return p

    def reset(self) -> None:
        self.resets += 1


def frame(index: int) -> bytes:
    """A frame whose every sample equals its own index, so PCM is traceable.

    Lets a test assert *which* frames landed in an utterance's audio, which is
    how the pre-roll and the contiguity of a max-length continuation are checked
    rather than assumed.
    """
    return np.full(cfg.FRAME_SAMPLES, index, dtype=np.int16).tobytes()


def frames_in(pcm: bytes) -> list[int]:
    """Recover the frame indices from PCM built by `frame()`."""
    block = np.frombuffer(pcm, dtype=np.int16).reshape(-1, cfg.FRAME_SAMPLES)
    assert (block == block[:, :1]).all(), "frames were spliced, not concatenated"
    return [int(v) for v in block[:, 0]]


def speech(n: int) -> list[float]:
    return [1.0] * n


def quiet(n: int) -> list[float]:
    return [0.0] * n


def segment(probabilities, config=None, on_speaking=None, during=None):
    """Run a probability script through a Segmenter; return it and its output."""
    segmenter = Segmenter(config or C, ScriptedVad(probabilities), on_speaking)
    out: list[Utterance] = []
    for i in range(len(probabilities)):
        utterance = segmenter.push(frame(i))
        if utterance is not None:
            out.append(utterance)
        if during is not None:
            during(i, segmenter)
    tail = segmenter.flush()
    if tail is not None:
        out.append(tail)
    return segmenter, out


def silero_probabilities(pcm: bytes, vad: SileroVad | None = None) -> np.ndarray:
    vad = vad or SileroVad()
    return np.array(
        [
            vad.probability(pcm[i : i + cfg.FRAME_BYTES])
            for i in range(0, len(pcm) - cfg.FRAME_BYTES + 1, cfg.FRAME_BYTES)
        ]
    )


@pytest.fixture(scope="module")
def fixture_pcm() -> bytes:
    """The generated speech clip, as raw 16 kHz mono int16."""
    with wave.open(str(cfg.SPEECH_FIXTURE_PATH), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
            cfg.SAMPLE_RATE,
            cfg.CHANNELS,
            cfg.SAMPLE_WIDTH_BYTES,
        )
        return wav.readframes(wav.getnframes())


def tone(hz: float, seconds: float, amp: int = 8_000) -> bytes:
    t = np.arange(int(cfg.SAMPLE_RATE * seconds)) / cfg.SAMPLE_RATE
    return (np.sin(2 * np.pi * hz * t) * amp).astype(np.int16).tobytes()


# --------------------------------------------------------------------------
# The contract — INTERFACES.md §2
# --------------------------------------------------------------------------


def test_utterance_has_exactly_the_documented_fields():
    """D11: `speaking` is a side channel and must not appear here."""
    assert list(Utterance.__dataclass_fields__) == [
        "id",
        "pcm",
        "start_ms",
        "end_ms",
        "closed_by",
        "preceding_silence_ms",
    ]


def test_silero_satisfies_the_detector_protocol():
    assert isinstance(SileroVad(), SpeechDetector)
    assert isinstance(ScriptedVad([]), SpeechDetector)


def test_a_wrong_sized_frame_is_refused():
    segmenter = Segmenter(C, ScriptedVad([1.0]))
    with pytest.raises(ValueError, match="1024-byte frame"):
        segmenter.push(b"\x00" * 1000)


# --------------------------------------------------------------------------
# Closing on silence (D12)
# --------------------------------------------------------------------------


def test_closes_on_silence_after_the_threshold():
    _, out = segment(quiet(10) + speech(50) + quiet(30))
    assert len(out) == 1
    u = out[0]
    assert u.closed_by == "silence"
    # Opens on frame 10, back-dated by the pre-roll; keeps the tail pad after
    # the last speech frame (59) and drops the rest of the silence.
    assert u.start_ms == (10 - PRE_ROLL_FRAMES) * FRAME_MS
    assert u.end_ms == (59 + 1 + TAIL_FRAMES) * FRAME_MS


def test_silence_shorter_than_the_threshold_does_not_close():
    """One utterance, not two: an 18-frame gap is under 600 ms."""
    _, out = segment(speech(30) + quiet(SILENCE_FRAMES - 1) + speech(30) + quiet(40))
    assert len(out) == 1
    assert out[0].closed_by == "silence"


def test_the_pre_roll_is_real_audio_from_before_the_trigger():
    """D55: Whisper must not be handed a clipped first phoneme."""
    _, out = segment(quiet(10) + speech(50) + quiet(30))
    assert frames_in(out[0].pcm)[0] == 10 - PRE_ROLL_FRAMES


def test_the_pre_roll_cannot_reach_before_the_start_of_the_session():
    _, out = segment(speech(50) + quiet(30))
    assert out[0].start_ms == 0
    assert frames_in(out[0].pcm)[0] == 0


def test_trailing_silence_beyond_the_tail_pad_is_dropped():
    """D55: what Whisper hallucinates "Thank you." onto (D30)."""
    _, out = segment(quiet(10) + speech(50) + quiet(40))
    kept_after_speech = frames_in(out[0].pcm)[-1] - 59
    assert kept_after_speech == TAIL_FRAMES


# --------------------------------------------------------------------------
# Closing on length (D12, D25) — PLAN.md: "no utterance exceeds max_utterance_ms"
# --------------------------------------------------------------------------


def test_no_utterance_exceeds_the_max():
    _, out = segment(speech(700))
    assert out, "continuous speech must still produce utterances"
    assert all(u.duration_ms <= C.max_utterance_ms for u in out)
    assert all(len(u.pcm) <= C.max_utterance_ms * BYTES_PER_MS for u in out)


def test_the_max_length_close_lands_exactly_on_the_cap():
    _, out = segment(speech(700))
    assert out[0].closed_by == "max_length"
    assert out[0].duration_ms == C.max_utterance_ms


def test_a_max_length_close_reopens_contiguously():
    """Level 5's fragment carry-over depends on the audio being unbroken."""
    _, out = segment(speech(700))
    assert out[1].preceding_silence_ms == 0
    assert out[1].start_ms == out[0].end_ms
    assert frames_in(out[1].pcm)[0] == frames_in(out[0].pcm)[-1] + 1


def test_a_speaker_who_never_pauses_still_produces_output():
    """The failure D12 exists to prevent: 22 s of speech, nothing on screen."""
    _, out = segment(speech(700))
    assert len(out) >= 5


# --------------------------------------------------------------------------
# The min_utterance_ms discard (D30)
# --------------------------------------------------------------------------


def test_a_sub_300_ms_blip_is_discarded_not_emitted():
    blip_frames = 3  # 96 ms
    segmenter, out = segment(quiet(5) + speech(blip_frames) + quiet(40))
    assert out == []
    assert segmenter.utterances_discarded == 1
    assert segmenter.utterances_emitted == 0


def test_the_discard_measures_speech_not_padding():
    """The padding is 320 ms — more than `min_utterance_ms` on its own.

    Testing `end_ms - start_ms` here would let every blip through and silently
    cancel D30's guard, because the pre-roll and tail pad alone exceed 300 ms.
    """
    blip = 3
    _, out = segment(quiet(5) + speech(blip) + quiet(40))
    assert out == []
    # Long enough to survive: 10 frames of speech is 320 ms.
    _, out = segment(quiet(5) + speech(10) + quiet(40))
    assert len(out) == 1
    assert out[0].duration_ms > C.min_utterance_ms  # padded, and that is fine


def test_a_discard_does_not_consume_an_id():
    _, out = segment(quiet(5) + speech(3) + quiet(40) + speech(50) + quiet(40))
    assert [u.id for u in out] == ["u_0001"]


def test_the_silence_gap_accumulates_across_a_discarded_blip():
    _, out = segment(quiet(5) + speech(3) + quiet(40) + speech(50) + quiet(40))
    # Measured from session start, not from the blip that was thrown away.
    assert out[0].preceding_silence_ms == out[0].start_ms


# --------------------------------------------------------------------------
# Invariants that must hold for every utterance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        quiet(10) + speech(50) + quiet(30),
        speech(700),
        quiet(5) + speech(3) + quiet(40) + speech(50) + quiet(40),
        quiet(10) + speech(40) + quiet(25) + speech(40) + quiet(25) + speech(300),
    ],
    ids=["single", "continuous", "blip-then-speech", "mixed"],
)
def test_invariants(script):
    _, out = segment(script)
    previous_end = 0
    for i, u in enumerate(out, start=1):
        assert u.id == f"u_{i:04d}"
        assert u.end_ms > u.start_ms
        assert u.closed_by in ("silence", "max_length")
        # The audio is exactly the span the timestamps claim.
        assert len(u.pcm) == u.duration_ms * BYTES_PER_MS
        # Reconstructible from the emitted fields alone.
        assert u.preceding_silence_ms == u.start_ms - previous_end
        assert u.duration_ms <= C.max_utterance_ms
        previous_end = u.end_ms


# --------------------------------------------------------------------------
# The runtime override (D28) — built ahead of its caller
# --------------------------------------------------------------------------


def test_the_override_shrinks_the_utterance_already_in_flight():
    def shrink_midway(i, segmenter):
        if i == 40:  # 1312 ms in, well before the 4000 ms cap
            segmenter.set_max_utterance_ms(2_000)

    _, out = segment(speech(700), during=shrink_midway)
    assert out[0].closed_by == "max_length"
    assert out[0].duration_ms == 1_984  # last whole frame at or under 2000 ms
    assert out[0].duration_ms < C.max_utterance_ms


def test_the_override_is_clamped_to_the_floor_and_the_configured_max():
    segmenter = Segmenter(C, ScriptedVad([]))
    assert segmenter.set_max_utterance_ms(100) == C.max_utterance_floor_ms
    assert segmenter.set_max_utterance_ms(99_999) == C.max_utterance_ms
    assert segmenter.set_max_utterance_ms(3_000) == 3_000
    assert segmenter.effective_max_utterance_ms == 3_000


def test_the_effective_max_starts_at_the_configured_value():
    assert Segmenter(C, ScriptedVad([])).effective_max_utterance_ms == C.max_utterance_ms


# --------------------------------------------------------------------------
# End of stream
# --------------------------------------------------------------------------


def test_flush_emits_the_utterance_still_open():
    """The last words of a session must not vanish."""
    _, out = segment(quiet(10) + speech(50))
    assert len(out) == 1
    assert out[0].closed_by == "silence"


def test_flush_when_nothing_is_open_returns_none():
    segmenter = Segmenter(C, ScriptedVad(quiet(10)))
    for i in range(10):
        segmenter.push(frame(i))
    assert segmenter.flush() is None


def test_flush_still_applies_the_discard():
    _, out = segment(quiet(10) + speech(3))
    assert out == []


# --------------------------------------------------------------------------
# The `speaking` side channel (D11)
# --------------------------------------------------------------------------


def test_speaking_fires_on_edges_only():
    edges: list[tuple[int, bool]] = []
    segmenter = Segmenter(C, ScriptedVad(quiet(10) + speech(50) + quiet(40)))
    segmenter.on_speaking = lambda v: edges.append((segmenter.frames_pushed - 1, v))
    for i in range(100):
        segmenter.push(frame(i))
    segmenter.flush()
    assert [v for _, v in edges] == [True, False]
    assert edges[0][0] == 10  # on with the first speech frame, not after it


def test_speaking_stays_dark_through_leading_silence():
    """A -1 sentinel put the indicator inside the debounce window at frame 0."""
    edges: list[bool] = []
    segmenter = Segmenter(C, ScriptedVad(quiet(30)), on_speaking=edges.append)
    for i in range(30):
        segmenter.push(frame(i))
        assert segmenter.speaking is False
    assert edges == []


def test_speaking_goes_dark_before_the_utterance_closes():
    """The dot needs 160 ms of confidence; the utterance needs 600."""
    segmenter = Segmenter(C, ScriptedVad(speech(30) + quiet(40)))
    closed_at = None
    dark_at = None
    for i in range(70):
        utterance = segmenter.push(frame(i))
        if dark_at is None and i >= 30 and not segmenter.speaking:
            dark_at = i
        if utterance is not None:
            closed_at = i
    assert dark_at is not None and closed_at is not None
    assert dark_at < closed_at


def test_a_short_dip_does_not_blink_the_indicator():
    """The debounce is 160 ms because a plosive dips for three frames."""
    segmenter = Segmenter(C, ScriptedVad(speech(20) + quiet(3) + speech(20)))
    for i in range(43):
        segmenter.push(frame(i))
        if i >= 1:
            assert segmenter.speaking is True


# --------------------------------------------------------------------------
# The Silero wrapper — real ONNX, real answers (D35, D53)
# --------------------------------------------------------------------------


def test_the_vad_reads_speech_high_and_silence_low(fixture_pcm):
    probabilities = silero_probabilities(fixture_pcm)
    assert probabilities.max() > 0.9
    assert probabilities[:15].max() < 0.1  # the fixture opens with 500 ms of silence


def test_the_vad_refuses_a_wrong_sized_frame():
    with pytest.raises(ValueError, match="1024-byte frame"):
        SileroVad().probability(b"\x00" * 512)


def test_context_is_carried_between_frames(fixture_pcm):
    """D53's guard. 64 samples of the previous frame precede the 512 new ones."""
    vad = SileroVad()
    assert vad.probability(fixture_pcm[: cfg.FRAME_BYTES]) is not None
    # The context after a frame is that frame's own tail, not zeros or the model
    # input's tail — the difference is what stops it degenerating.
    tail = np.frombuffer(fixture_pcm[:cfg.FRAME_BYTES], dtype=np.int16)[
        -cfg.VAD_CONTEXT_SAMPLES:
    ].astype(np.float32) / 32768.0
    assert np.allclose(vad._context, tail)
    assert silero_probabilities(fixture_pcm).max() > 0.9


def test_control_dropping_the_context_kills_the_vad(fixture_pcm, monkeypatch):
    """The control for the test above: without the context, speech reads as silence.

    This is the bug exactly as it shipped — a 512-wide call runs cleanly and
    returns a plausible float. Without this control, a guard asserting "> 0.9"
    could be satisfied by any model at all and nobody would know the difference.
    """
    vad = SileroVad()
    zeros = np.zeros(cfg.VAD_CONTEXT_SAMPLES, dtype=np.float32)
    original = vad._session.run

    def run_without_context(_outputs, inputs):
        inputs = dict(inputs)
        inputs["input"] = inputs["input"][:, cfg.VAD_CONTEXT_SAMPLES :]
        return original(_outputs, inputs)

    monkeypatch.setattr(vad._session, "run", run_without_context)
    assert np.allclose(vad._context, zeros)
    broken = silero_probabilities(fixture_pcm, vad)
    # 0.05 against 1.00: the model is not merely less confident, it is answering
    # "no speech" to speech. Nothing reaches the 0.50 open threshold, so the
    # whole pipeline downstream of here would produce silence and no error.
    assert broken.max() < C.vad_speech_threshold / 5, (
        f"the 512-wide call peaked at {broken.max():.4f} on speech; the guard "
        f"above cannot distinguish a working VAD from a dead one"
    )


def test_lstm_state_is_carried_between_frames(fixture_pcm):
    vad = SileroVad()
    with_state = silero_probabilities(fixture_pcm, vad)
    assert not np.allclose(vad._state, 0.0)

    stateless = SileroVad()
    per_frame = []
    for i in range(0, len(fixture_pcm) - cfg.FRAME_BYTES + 1, cfg.FRAME_BYTES):
        stateless.reset()  # the bug: forget everything each frame
        per_frame.append(stateless.probability(fixture_pcm[i : i + cfg.FRAME_BYTES]))

    # The control: if resetting changed nothing, the state would not be state.
    assert not np.allclose(with_state, per_frame, atol=0.05)


def test_reset_returns_the_vad_to_its_opening_state(fixture_pcm):
    vad = SileroVad()
    first = vad.probability(fixture_pcm[: cfg.FRAME_BYTES])
    silero_probabilities(fixture_pcm, vad)
    vad.reset()
    assert vad.probability(fixture_pcm[: cfg.FRAME_BYTES]) == pytest.approx(first)


def test_the_vad_never_touches_the_gpu():
    """4 GB is the ceiling and CTranslate2 gets it (D18)."""
    assert SileroVad()._session.get_providers() == ["CPUExecutionProvider"]


def test_the_vad_cost_is_near_the_measured_budget():
    """PLAN.md: "VAD cost stays near the measured 0.156 ms/frame" (D35)."""
    vad = SileroVad()
    silent = b"\x00" * cfg.FRAME_BYTES
    for _ in range(20):
        vad.probability(silent)
    started = time.perf_counter()
    n = 200
    for _ in range(n):
        vad.probability(silent)
    per_frame_ms = (time.perf_counter() - started) * 1000 / n
    # A generous ceiling: the point is that it stays in the "free" order of
    # magnitude on a shared CPU, not that this laptop's exact number reappears.
    assert per_frame_ms < 1.0, f"{per_frame_ms:.3f} ms per {FRAME_MS} ms frame"


# --------------------------------------------------------------------------
# End to end — PLAN.md's acceptance criteria through the real VAD
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "pcm"),
    [
        ("silence", b"\x00" * cfg.FRAME_BYTES * 300),
        ("tone", tone(440.0, 6.0)),
        ("chord", tone(261.6, 6.0) + tone(329.6, 6.0) + tone(392.0, 6.0)),
        ("noise", np.random.default_rng(0).normal(0, 3000, 96_000).astype(np.int16).tobytes()),
    ],
)
def test_non_speech_produces_zero_utterances(label, pcm):
    """PLAN.md: silence-only and music-only input produce zero utterances.

    This is the criterion that pays for choosing Silero over an energy-and-
    spectrum heuristic (D23): every false positive here is a wasted Whisper call
    on the scarce resource plus a hallucinated caption on screen.
    """
    segmenter = Segmenter(C, SileroVad())
    utterances = list(
        segmenter.utterances(
            pcm[i : i + cfg.FRAME_BYTES]
            for i in range(0, len(pcm) - cfg.FRAME_BYTES + 1, cfg.FRAME_BYTES)
        )
    )
    assert utterances == [], f"{label} produced {len(utterances)} utterance(s)"


def test_the_fixture_segments_at_its_known_pauses():
    """The fixture's silences are 900, 900 and 1000 ms by construction (D56)."""
    segmenter = Segmenter(C, SileroVad())
    out = list(segmenter.utterances(WavFileSource(cfg.SPEECH_FIXTURE_PATH).frames()))

    assert [u.closed_by for u in out] == [
        "silence",
        "silence",
        "max_length",   # the third phrase is deliberately longer than 4000 ms
        "silence",
    ]
    # Each boundary lands inside the pause that was inserted there. The measured
    # gap is not exactly 900 ms and should not be: the pre-roll and tail pad eat
    # 320 ms of it, and Silero's own onset and release lag give some back. What
    # matters is that a gap appears where a gap was put, and that it never runs
    # longer than the silence itself — which would mean speech was cut.
    for gap in (out[1].preceding_silence_ms, out[2].preceding_silence_ms):
        assert 900 - 350 <= gap <= 900
    assert out[3].preceding_silence_ms == 0  # the max-length continuation

    previous_end = 0
    for u in out:
        assert len(u.pcm) == u.duration_ms * BYTES_PER_MS
        assert u.duration_ms <= C.max_utterance_ms
        assert u.preceding_silence_ms == u.start_ms - previous_end
        previous_end = u.end_ms


def test_a_shrunk_max_cuts_the_fixture_into_more_utterances():
    """The D28 seam, exercised end to end rather than only against a fake."""
    baseline = Segmenter(C, SileroVad())
    normal = list(baseline.utterances(WavFileSource(cfg.SPEECH_FIXTURE_PATH).frames()))

    shrunk = Segmenter(C, SileroVad())
    shrunk.set_max_utterance_ms(C.max_utterance_floor_ms)
    tighter = list(shrunk.utterances(WavFileSource(cfg.SPEECH_FIXTURE_PATH).frames()))

    assert len(tighter) > len(normal)
    assert all(u.duration_ms <= C.max_utterance_floor_ms for u in tighter)


# --------------------------------------------------------------------------
# tools/dump_utterances
# --------------------------------------------------------------------------


def test_dump_utterances_end_to_end(tmp_path, capsys):
    from speech_translator.tools import dump_utterances

    code = dump_utterances.main(
        [
            "--input-wav",
            str(cfg.SPEECH_FIXTURE_PATH),
            "--write-wav",
            str(tmp_path),
            "--json",
        ]
    )
    assert code == 0

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert len(rows) == 4
    assert {r["closed_by"] for r in rows} == {"silence", "max_length"}

    # Each utterance is on disk and playable — the actual answer to "did the
    # boundaries land at real pauses".
    for row in rows:
        with wave.open(row["wav"], "rb") as wav:
            assert wav.getframerate() == cfg.SAMPLE_RATE
            assert wav.getnchannels() == cfg.CHANNELS
            assert wav.getnframes() == row["bytes"] // cfg.SAMPLE_WIDTH_BYTES


def test_dump_utterances_reports_the_vad_cost_and_totals():
    from speech_translator.tools import dump_utterances

    summary = dump_utterances.dump(input_wav=str(cfg.SPEECH_FIXTURE_PATH), config=C)
    assert summary["emitted"] == 4
    assert 0.0 < summary["vad_ms_per_frame"] < 1.0
    assert 0.5 < summary["speech_ratio"] < 0.9
    assert "BUG" not in dump_utterances.render(summary)


def test_dump_utterances_honours_the_runtime_override():
    from speech_translator.tools import dump_utterances

    summary = dump_utterances.dump(
        input_wav=str(cfg.SPEECH_FIXTURE_PATH),
        max_utterance_ms=C.max_utterance_floor_ms,
        config=C,
    )
    assert summary["effective_max_utterance_ms"] == C.max_utterance_floor_ms
    assert all(
        u["duration_ms"] <= C.max_utterance_floor_ms for u in summary["utterances"]
    )


def test_dump_utterances_says_so_when_there_are_no_utterances():
    from speech_translator.tools import dump_utterances

    summary = {
        "utterances": [], "frames": 0, "audio_ms": 0, "speech_ms": 0,
        "speech_ratio": 0.0, "emitted": 0, "discarded": 0,
        "effective_max_utterance_ms": 4000, "vad_ms_per_frame": 0.0,
        "wall_clock_s": 0.0, "interrupted": False, "device_lost": None,
    }
    assert "no utterances" in dump_utterances.render(summary)


# --------------------------------------------------------------------------
# Config — the two TUNABLE values this level added (D54, D55)
# --------------------------------------------------------------------------


def test_the_thresholds_have_hysteresis():
    assert C.vad_release_threshold < C.vad_speech_threshold


def test_a_single_threshold_would_chatter_on_a_real_falling_edge():
    """The measured edge that motivated two thresholds: 1.00, 0.98, 0.74, 0.11."""
    falling = [1.00, 0.98, 0.74, 0.11]
    single = [p >= C.vad_speech_threshold for p in falling]
    assert single == [True, True, True, False]  # 0.74 survives a 0.5 cut anyway

    # The one that matters is a dip *below* 0.5 that hysteresis holds through.
    dip = [1.00, 0.44, 0.41, 0.98]
    hysteresis, on = [], False
    for p in dip:
        on = p >= (C.vad_release_threshold if on else C.vad_speech_threshold)
        hysteresis.append(on)
    assert hysteresis == [True, True, True, True]
    assert [p >= C.vad_speech_threshold for p in dip] == [True, False, False, True]


def test_the_padding_is_smaller_than_the_silence_threshold():
    """The tail pad must not eat into the gap that closed the utterance."""
    assert C.utterance_tail_pad_ms < C.silence_threshold_ms
    assert C.utterance_pre_roll_ms % FRAME_MS == 0
    assert C.utterance_tail_pad_ms % FRAME_MS == 0


def test_the_d25_values_are_untouched_by_this_level():
    assert (C.silence_threshold_ms, C.max_utterance_ms) == (600, 4_000)
    assert (C.min_utterance_ms, C.max_utterance_floor_ms) == (300, 2_000)


def test_the_tunables_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("ST_VAD_SPEECH_THRESHOLD", "0.7")
    monkeypatch.setenv("ST_UTTERANCE_PRE_ROLL_MS", "64")
    c = cfg.load_config()
    assert c.vad_speech_threshold == 0.7
    assert c.utterance_pre_roll_ms == 64


def test_a_retuned_threshold_reaches_the_segmenter():
    """Config is read at construction, not baked in at import."""
    strict = dataclasses.replace(C, vad_speech_threshold=0.99, vad_release_threshold=0.99)
    _, out = segment(quiet(5) + [0.9] * 50 + quiet(40), config=strict)
    assert out == []
