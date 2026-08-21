# Session 10 — Level 4: transcription worker

_2026-08-21. Linux dev box. Claude Code session._

## What happened

Level 3 selected medium:int8_float16. Level 4 builds the transcription worker
process that turns `Utterance` objects into `Transcript` objects: a subprocess
owned by `TranscribeWorker`, talking over two `multiprocessing.Queue` instances.

## Files created

```
speech_translator/transcribe/
    __init__.py         public re-exports
    protocol.py         Transcript + IPC message types (WorkerReady/Result/Error)
    guard.py            is_accepted() — hallucination guard (D30)
    lid.py              LidAccumulator — windowed language ID (D32)
    worker.py           _load_model, _run_worker_loop, worker_main, TranscribeWorker
tests/test_transcribe.py   24 tests, all pass on Linux with no GPU
```

## Key design choices

### Three-layer structure (D61)
`worker_main` is split so `_run_worker_loop` can be called from tests directly
with a `FakeWhisperModel` and a stdlib `queue.Queue`. This follows the same
injection discipline as `SpeechDetector` in Level 2 (D56). Without it, every
test would require spawning a subprocess and a CUDA context.

`FakeWhisperModel.transcribe()` returns a *lazy generator* (not a list), so
`test_worker_loop_drains_generator` can detect if the loop forgot to drain it.
This is the D58 lesson applied one layer up.

### Hallucination guard (D30)
`is_accepted(no_speech_prob, avg_logprob, config)` in `guard.py`. Pure function,
no state. Aggregation across segments: max of `no_speech_prob` (pessimistic —
any silent segment condemns the utterance), mean of `avg_logprob` (each segment
is already a token-level average). Empty segments default to 1.0 / −∞ and are
rejected (D63).

### Windowed LID (D32)
`LidAccumulator` in `lid.py` uses `config.lid_window_speech_ms` (10 000 ms).
Accumulates confidence-weighted votes per utterance; locks when cumulative speech
reaches the window. After locking, `kwargs["language"]` is pinned for all
subsequent `model.transcribe()` calls. LID updates on rejected utterances too
(D64 — the language evidence in `info` is independent of text quality).

### Worker lifecycle
`TranscribeWorker.start()` uses `multiprocessing.get_context("spawn")` (required
on Windows — D27) and blocks on `out_queue.get(timeout=120)` for either
`WorkerReady` or `WorkerError`. Load failure reaches the caller immediately as
`worker.state == "error"` rather than a 120-second hang.

### JSONL log (D34)
Written to `config.logs_dir / asr_<timestamp>.jsonl` (one file per worker
session). Fields: `utterance_id, audio_ms, asr_ms, rtf, language,
no_speech_prob, avg_logprob, accepted`. This is the evidence base for the
Level 7 latency measurements.

## What the tests cover (Linux, no GPU)

| Test | What it proves |
|---|---|
| `test_guard_*` (5) | boundary and both rejection conditions |
| `test_lid_*` (8) | accumulation, lock timing, weighted winner, reset |
| `test_worker_loop_produces_transcript` | valid → WorkerResult with Transcript |
| `test_worker_loop_rejects_hallucination` | nsp=0.95 → WorkerResult(None) |
| `test_worker_loop_rejects_low_logprob` | alp=−2.0 → rejected |
| `test_worker_loop_rejects_empty_text` | whitespace → rejected |
| `test_worker_loop_shutdown` | None sentinel exits cleanly |
| `test_worker_loop_multiple_utterances` | accept/reject/accept sequence |
| `test_worker_loop_drains_generator` | side effect confirms generator consumed |
| `test_worker_loop_jsonl_log` | required fields, rtf is float |
| `test_worker_loop_lid_pins_language` | 4th call gets language= kwarg |
| `test_worker_loop_transcript_carries_lid_info` | language + confidence in Transcript |
| `test_worker_main_load_failure_sends_error` | monkeypatched _load_model → WorkerError |
| `test_protocol_picklable` | all IPC types survive pickle round-trip |

## What is owed to Windows

- `TranscribeWorker.start()` with a real GPU — this is Level 7's sustained run.
- WAV → correct punctuated text → correct language detected (acceptance criterion 1).
- Per-utterance RTF in the JSONL log during a real session (criterion 4).

## Pre-existing Windows test issue (not caused here)

`test_segment.py` has 7 parametrised tests that fail with `ValueError: the
environment variable is longer than 32767 characters`. pytest sets
`PYTEST_CURRENT_TEST` to the full test ID; test IDs embed raw PCM bytes from the
parametrize call, producing a ~30 KB string. Windows caps env vars at 32767 chars.
This predates Level 4 and is unrelated to the transcribe package. All other suites
pass on Windows.
