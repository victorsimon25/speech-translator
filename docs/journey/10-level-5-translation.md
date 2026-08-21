# Session 11 — Level 5: Sentences + Translation

**Date:** 2026-08-21
**Branch:** build
**Outcome:** Level 5 complete. All 8 acceptance tests pass. No regressions.

---

## What was built

`speech_translator/translate/` package with five files:

| File | What it does |
|---|---|
| `protocol.py` | `Sentence` and `Translation` frozen dataclasses (INTERFACES.md §4–5) |
| `splitter.py` | `SentenceSplitter` — fragment carry-over, `.?!` split, flush rule |
| `translator.py` | `Translator` — REST call, LRU cache, monthly budget persistence |
| `task.py` | `run_translation_task` — single serialised asyncio consumer (D27) |
| `__init__.py` | Package exports |

---

## Key decisions made

**D66 (see DECISIONS.md):**

1. Monthly budget file: `data_dir/usage/chars_YYYY_MM.json` rather than the
   single `mt_budget_file` already in Config. The per-month key is load-bearing
   because the quota does not roll over across months.

2. `feed()` takes `preceding_silence_ms` and `closed_by` as keyword params.
   `Transcript` (transcribe/protocol.py) does not carry these — they are stripped
   when the worker emits a result. The main pipeline passes them from the parent
   `Utterance`. Safe defaults allow callers that only hold a Transcript to work.

3. `Translator` accepts an optional `Config` parameter for testability. Without
   it, the budget-persistence test cannot redirect file I/O to `tmp_path`.

---

## Test results

```
tests/test_translate.py::test_splitter_carry_over              PASSED
tests/test_translate.py::test_splitter_flush_emits_held_fragment PASSED
tests/test_translate.py::test_splitter_preceding_silence_ms    PASSED
tests/test_translate.py::test_translation_ordering             PASSED
tests/test_translate.py::test_translation_budget_persists      PASSED
tests/test_translate.py::test_translation_api_failure_degrades PASSED
tests/test_translate.py::test_translation_skip_same_lang       PASSED
tests/test_translate.py::test_translation_lru_cache            PASSED

8 passed in 1.40s
```

Full suite: no regressions in test_audio, test_benchmark, test_foundation,
test_transcribe. The 8 `test_segment.py` setup errors are pre-existing (PCM
binary data embedded in pytest IDs exceeds 32767 chars on Windows; not caused
by this level).

---

## What's next

**Level 6 — Server + UI.** FastAPI + uvicorn, six-state session controller,
WebSocket push to the browser, caption cards in vanilla HTML/CSS/JS (D38).
