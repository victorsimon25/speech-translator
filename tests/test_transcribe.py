"""Tests for speech_translator.transcribe.

All tests run on Linux with no GPU, no audio hardware, no network.
The WhisperModel is replaced by FakeWhisperModel (injection pattern from D56/D58).
"""

from __future__ import annotations

import json
import pickle
import queue
from typing import Iterator

import pytest

import speech_translator.config as cfg
from speech_translator.segment import BYTES_PER_MS, Utterance
from speech_translator.transcribe.guard import is_accepted
from speech_translator.transcribe.lid import LidAccumulator
from speech_translator.transcribe.protocol import (
    Transcript,
    WorkerError,
    WorkerReady,
    WorkerResult,
)
from speech_translator.transcribe.worker import _run_worker_loop, worker_main


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeSegment:
    def __init__(
        self,
        text: str,
        no_speech_prob: float = 0.01,
        avg_logprob: float = -0.3,
    ) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class FakeInfo:
    def __init__(self, language: str = "es", language_probability: float = 0.99) -> None:
        self.language = language
        self.language_probability = language_probability


class FakeWhisperModel:
    """Stand-in for WhisperModel.  Returns a lazy generator (D58 lesson)."""

    def __init__(self, responses: list[tuple[str, float, float]]) -> None:
        self._it = iter(responses)
        self.last_kwargs: dict = {}

    def transcribe(self, audio, **kwargs) -> tuple[Iterator[FakeSegment], FakeInfo]:
        self.last_kwargs = dict(kwargs)
        text, nsp, alp = next(self._it)

        def _gen() -> Iterator[FakeSegment]:
            yield FakeSegment(text, nsp, alp)

        return _gen(), FakeInfo()


def make_utterance(
    id_: str = "u_0001",
    speech_ms: int = 300,
    start_ms: int = 0,
) -> Utterance:
    pcm = b"\x00" * (speech_ms * BYTES_PER_MS)
    return Utterance(
        id=id_,
        pcm=pcm,
        start_ms=start_ms,
        end_ms=start_ms + speech_ms,
        closed_by="silence",
        preceding_silence_ms=0,
    )


def _run(model, utterances, *, log_path=None):
    """Drive _run_worker_loop with a list of utterances; return all WorkerResults."""
    config = cfg.load_config()
    lid = LidAccumulator(config)
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    for u in utterances:
        in_q.put(u)
    in_q.put(None)
    _run_worker_loop(in_q, out_q, model, lid, config, log_path)
    results = []
    while not out_q.empty():
        results.append(out_q.get_nowait())
    return results


# ---------------------------------------------------------------------------
# Hallucination guard (D30)
# ---------------------------------------------------------------------------


def test_guard_accepts_clean():
    config = cfg.load_config()
    assert is_accepted(0.01, -0.3, config)


def test_guard_rejects_no_speech():
    config = cfg.load_config()
    assert not is_accepted(0.95, -0.3, config)


def test_guard_rejects_low_logprob():
    config = cfg.load_config()
    assert not is_accepted(0.01, -2.0, config)


def test_guard_boundary_values():
    """Values exactly at the threshold are accepted (not strict inequality)."""
    config = cfg.load_config()
    assert is_accepted(config.no_speech_prob_max, config.avg_logprob_min, config)


def test_guard_both_bad():
    config = cfg.load_config()
    assert not is_accepted(0.99, -3.0, config)


# ---------------------------------------------------------------------------
# Language identification accumulator (D32)
# ---------------------------------------------------------------------------


def test_lid_no_lock_below_window():
    config = cfg.load_config()
    lid = LidAccumulator(config)
    for _ in range(3):
        lid.update("es", 0.99, 2_000)  # 3 × 2 s = 6 s < 10 s
    assert lid.locked_language is None


def test_lid_locks_at_window():
    config = cfg.load_config()
    lid = LidAccumulator(config)
    for _ in range(3):
        lid.update("es", 0.99, 4_000)  # 3 × 4 s = 12 s ≥ 10 s → lock on 3rd
    assert lid.locked_language == "es"


def test_lid_picks_weighted_winner():
    """Language with higher total weighted confidence wins the lock."""
    config = cfg.load_config()
    lid = LidAccumulator(config)
    # "es" gets 2 × 0.90 = 1.80; "en" gets 1 × 0.99 = 0.99 — "es" should win
    lid.update("es", 0.90, 4_000)
    lid.update("en", 0.99, 4_000)
    lid.update("es", 0.90, 4_000)  # total = 12 s → lock
    assert lid.locked_language == "es"


def test_lid_no_update_after_lock():
    """Further calls after lock do not change the locked language."""
    config = cfg.load_config()
    lid = LidAccumulator(config)
    for _ in range(3):
        lid.update("es", 0.99, 4_000)
    lid.update("fr", 0.99, 4_000)
    assert lid.locked_language == "es"


def test_lid_best_language_before_lock():
    config = cfg.load_config()
    lid = LidAccumulator(config)
    lid.update("es", 0.9, 1_000)
    lang, conf = lid.best_language()
    assert lang == "es"
    assert conf == pytest.approx(1.0)


def test_lid_best_language_none_before_any_update():
    lid = LidAccumulator()
    assert lid.best_language() is None


def test_lid_reset():
    config = cfg.load_config()
    lid = LidAccumulator(config)
    for _ in range(3):
        lid.update("es", 0.99, 4_000)
    assert lid.locked_language == "es"
    lid.reset()
    assert lid.locked_language is None
    assert lid.best_language() is None


# ---------------------------------------------------------------------------
# Worker loop — core logic (D30, D32, D34)
# ---------------------------------------------------------------------------


def test_worker_loop_produces_transcript():
    model = FakeWhisperModel([("Hola, ¿cómo están?", 0.01, -0.3)])
    results = _run(model, [make_utterance()])
    assert len(results) == 1
    assert isinstance(results[0], WorkerResult)
    assert results[0].transcript is not None
    assert results[0].transcript.text == "Hola, ¿cómo están?"


def test_worker_loop_rejects_hallucination():
    """High no_speech_prob → None transcript rather than a caption (D30)."""
    model = FakeWhisperModel([("Thank you.", 0.95, -0.2)])
    results = _run(model, [make_utterance()])
    assert results[0].transcript is None


def test_worker_loop_rejects_low_logprob():
    model = FakeWhisperModel([("mumble", 0.05, -3.0)])
    results = _run(model, [make_utterance()])
    assert results[0].transcript is None


def test_worker_loop_rejects_empty_text():
    """Whisper returning only whitespace is treated as no output."""
    model = FakeWhisperModel([("   ", 0.05, -0.3)])
    results = _run(model, [make_utterance()])
    assert results[0].transcript is None


def test_worker_loop_shutdown():
    """None sentinel causes the loop to exit cleanly with no result."""
    model = FakeWhisperModel([])
    results = _run(model, [])
    assert results == []


def test_worker_loop_multiple_utterances():
    model = FakeWhisperModel(
        [("Hola.", 0.01, -0.3), ("Thank you.", 0.95, -0.2), ("Adiós.", 0.01, -0.4)]
    )
    utterances = [make_utterance(id_=f"u_{i:04d}", start_ms=i * 500) for i in range(3)]
    results = _run(model, utterances)
    assert len(results) == 3
    transcripts = [r.transcript for r in results]
    assert transcripts[0] is not None and transcripts[0].text == "Hola."
    assert transcripts[1] is None  # hallucination guard
    assert transcripts[2] is not None and transcripts[2].text == "Adiós."


def test_worker_loop_drains_generator():
    """The segments generator must be consumed inside _run_worker_loop (D58).

    If it were drained by the caller after the timed region, the RTF would be
    meaninglessly close to zero — the transcription work happens in the generator.
    """
    drained: list[bool] = []

    class SideEffectModel:
        def transcribe(self, audio, **kwargs):
            def _gen():
                drained.append(True)
                yield FakeSegment("Hola.", 0.01, -0.3)

            return _gen(), FakeInfo()

    _run(SideEffectModel(), [make_utterance()])
    assert drained == [True], "Generator was not consumed inside _run_worker_loop"


def test_worker_loop_jsonl_log(tmp_path):
    """JSONL log contains one record per utterance with all required fields (D34)."""
    log_path = tmp_path / "asr_test.jsonl"
    model = FakeWhisperModel([("Hola.", 0.01, -0.3)])

    config = cfg.load_config()
    lid = LidAccumulator(config)
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    in_q.put(make_utterance(speech_ms=500))
    in_q.put(None)
    _run_worker_loop(in_q, out_q, model, lid, config, log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    required = {"utterance_id", "audio_ms", "asr_ms", "rtf", "language",
                "no_speech_prob", "avg_logprob", "accepted"}
    assert required <= record.keys()
    assert record["audio_ms"] == 500
    assert record["accepted"] is True
    assert isinstance(record["rtf"], float)


def test_worker_loop_lid_pins_language():
    """After the LID window closes, 'language' kwarg is passed to transcribe (D32).

    With lid_window_speech_ms=10 000 and 4 utterances of 4 000 ms each:
      call 1: no pin (total=4 000 ms)
      call 2: no pin (total=8 000 ms)
      call 3: no pin (total=12 000 ms → locks AFTER this call)
      call 4: pinned to "es"
    """
    config = cfg.load_config()
    lid = LidAccumulator(config)

    class TrackingModel:
        def __init__(self):
            self.calls: list[dict] = []

        def transcribe(self, audio, **kwargs):
            self.calls.append(dict(kwargs))

            def _gen():
                yield FakeSegment("Hola.", 0.01, -0.3)

            return _gen(), FakeInfo("es", 0.99)

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    model = TrackingModel()

    for i in range(4):
        in_q.put(make_utterance(id_=f"u_{i:04d}", speech_ms=4_000, start_ms=i * 4_000))
    in_q.put(None)
    _run_worker_loop(in_q, out_q, model, lid, config, None)

    assert "language" not in model.calls[0]
    assert "language" not in model.calls[1]
    assert "language" not in model.calls[2]
    assert model.calls[3].get("language") == "es"


def test_worker_loop_transcript_carries_lid_info():
    """Transcript.language and language_confidence come from the model's info."""
    model = FakeWhisperModel([("Buenos días.", 0.01, -0.3)])
    results = _run(model, [make_utterance()])
    t = results[0].transcript
    assert t is not None
    assert t.language == "es"
    assert t.language_confidence == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# worker_main error path
# ---------------------------------------------------------------------------


def test_worker_main_load_failure_sends_error(monkeypatch):
    """worker_main puts WorkerError to out_queue when model load fails."""
    import speech_translator.transcribe.worker as wmod

    def fail_load(config):
        raise RuntimeError("CUDA not available — injected for test")

    monkeypatch.setattr(wmod, "_load_model", fail_load)

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    worker_main(in_q, out_q, cfg.load_config())

    msg = out_q.get(timeout=1)
    assert isinstance(msg, WorkerError)
    assert "CUDA" in msg.message
    assert "injected for test" in msg.message
    assert "RuntimeError" in msg.detail  # traceback text


# ---------------------------------------------------------------------------
# Protocol pickling (IPC contract)
# ---------------------------------------------------------------------------


def test_protocol_picklable():
    """All IPC message types survive a pickle round-trip (multiprocessing.Queue)."""
    transcript = Transcript(
        utterance_id="u_0001",
        text="Hola.",
        language="es",
        language_confidence=0.99,
        asr_ms=245,
        no_speech_prob=0.01,
        avg_logprob=-0.3,
    )
    objects = [
        WorkerReady(),
        WorkerResult(transcript=transcript),
        WorkerResult(transcript=None),
        WorkerError(message="oops", detail="traceback here"),
        transcript,
    ]
    for obj in objects:
        assert pickle.loads(pickle.dumps(obj)) == obj
