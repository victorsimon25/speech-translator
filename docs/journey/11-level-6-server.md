# Session 12 — Level 6: Server + UI

_Date: 2026-08-21_

## What was built

Level 6 wires the pipeline (Levels 0–5) into a running web server with a browser caption UI:

- **`speech_translator/server/session.py`** — `SessionController`, the asyncio state machine. Six states: idle → loading → running ↔ degraded → error. Owns the pipeline lifecycle: starts/stops the `TranscribeWorker`, `Segmenter`, `SentenceSplitter`, translation task, and `Publisher`. Accepts a `worker_factory` parameter so tests can inject `FakeTranscribeWorker` without touching CUDA.

- **`speech_translator/server/backpressure.py`** — `BackpressureController` (D28). Tracks consecutive utterances where queue depth ≥ `shrink_trigger_depth`; after `shrink_after_utterances` consecutive hits it calls `segmenter.set_max_utterance_ms(int(current * shrink_factor))` and transitions to "degraded". When queue depth returns to 0 and RTF EMA drops below `recover_rtf_ema` for `recover_after_s` seconds, it grows back and transitions to "running". Drop-oldest path: if the ASR queue is full, `get_nowait()` the front item before `put_nowait()`-ing the new one.

- **`speech_translator/server/publisher.py`** — `Publisher`. Maintains a `dict[sentence_id, Sentence]` registry. When a `Translation` arrives, `pop(sentence_id)` to retrieve the matching `Sentence`, then emit a `{"type":"caption"}` message with `utterance_id`, `preceding_silence_ms`, `closed_by`, latency breakdown, and both source/target texts. Health message every 2 s: `queue_depth`, `rtf`, `word_age_ms`, `effective_max_utterance_ms`, `chars_used`, `chars_budget`.

- **`speech_translator/server/app.py`** — `create_app()`. `GET /` → `index.html`, `StaticFiles` at `/static`, `WS /ws`. One WebSocket client at a time (D38). Session and broadcast are scoped to the app instance via closure.

- **`speech_translator/server/fake.py`** — `FakeTranscribeWorker`. Satisfies the same interface as `TranscribeWorker` (`.start()`, `.stop()`, `.in_queue`, `.out_queue`, `.state`, `.last_error`). `start()` sets `state="ready"` and spawns a daemon thread; does **not** put `WorkerReady` on `out_queue` (the real worker drains it internally, so the bridge task never sees it — the fake must match).

- **`speech_translator/server/static/`** — `index.html` (status badge, language row, health bar, captions container, jump-to-live button, export), `app.js` (vanilla WebSocket client, append-only caption cards, gap dividers at ≥1200 ms silence, auto-scroll, LID chip + click-to-override, Blob export), `style.css` (six state colours: idle=grey, loading=blue pulse, listening=yellow, running=green, degraded=orange, error=red).

- **`speech_translator/__main__.py`** updated from stub to: `uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)`.

- **`Translator.chars_used` property** added to expose the monthly counter for the health message.

## Key engineering decision: the async/sync bridge (D67)

The pipeline's `TranscribeWorker` puts results on a `multiprocessing.Queue` from a subprocess. The server runs on asyncio. Bridging these cleanly turned out to be the hardest part of Level 6.

**First attempt: blocking `queue.get()` in an executor.**

```python
result = await loop.run_in_executor(None, worker.out_queue.get)
```

This works while the pipeline runs. But when `_stop()` calls `task.cancel()`, the `CancelledError` cannot be delivered because the executor thread is blocked on `queue.get()` with no timeout. `asyncio.run()` then hangs waiting for the thread pool to drain (default 300 s timeout on Python 3.12).

**Fix: three-part unblocking protocol.**

1. `out_queue.get(timeout=0.5)` — the thread exits within one timeout window even if `task.cancel()` arrives in the gap. `CancelledError` is delivered at the `await asyncio.sleep(0)` that follows each empty-queue retry.

2. `_stop()` puts `None` on `out_queue` before calling `task.cancel()`. This unblocks the thread immediately rather than waiting up to 0.5 s.

3. The bridge treats `None` as a shutdown sentinel: `if result is None: break`.

**FakeTranscribeWorker contract.**

An earlier version of `fake.py` put `WorkerReady()` on `out_queue` in `start()` — mimicking what the real worker does internally. But the real `TranscribeWorker.start()` calls `out_queue.get(timeout=120)` itself before returning, so the bridge task never sees `WorkerReady`. The fake must not put it there either, or the bridge receives an unexpected type and the state machine diverges from the real path. Removed, and the contract is now documented in the class docstring.

## Tests (all 6 pass on Linux, no GPU, no audio, no network)

| Test | What it checks |
|---|---|
| `test_state_machine_transitions` | idle → loading → running → idle via asyncio.run |
| `test_caption_join` | Publisher joins Translation+Sentence; `preceding_silence_ms` and `closed_by` come from Sentence |
| `test_backpressure_shrink` | queue depth at threshold for N utterances → `effective_max_utterance_ms` shrinks |
| `test_backpressure_drop_oldest` | full queue → oldest dropped, dropped event emitted, correct items remain |
| `test_websocket_state_broadcast` | state transitions broadcast over WebSocket (Starlette TestClient) |
| `test_health_message_fields` | all 7 fields present with correct types |

Full suite: 133 passed, 4 skipped. The 4 skips are the pre-existing Windows-only `test_segment.py` issue (pytest env-var exceeds 32767 chars on parametrised tests with binary PCM IDs) — not caused by Level 6.

## Decisions appended

**D67** — Publisher sentence registry design; `_asr_bridge` unblocking mechanism; `FakeTranscribeWorker` contract.
