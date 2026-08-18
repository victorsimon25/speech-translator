# Speech Translator

Real-time speech translator for online meetings. Captures **system audio**
(what the speakers are playing), transcribes it locally, translates it, and
shows captions in the browser.

## Stack
Python 3.12 backend + browser UI over WebSocket. Local Whisper on **CUDA** for
transcription, Google Cloud Translation (free tier) for translation.

## Where this runs
- **Target platform: Windows.** GTX 1650 Ti, 4 GB VRAM. Linux is descoped to the
  `AudioSource` interface only (D17).
- **Written on Linux, tested on Windows**, with git as the bridge (D22). Code
  must import cleanly on Linux: Windows-only imports are lazy, and
  platform-specific deps carry environment markers.

## Read before working
1. `docs/STATE.md` — what's done, in progress, next, blocked. **Read first.**
2. `docs/INTERFACES.md` — module contracts. Read the stage you're touching;
   you should not need to read another stage's source.
3. `docs/DECISIONS.md` — why things are the way they are. Check before
   proposing a change.

## Hard rules
- Free tiers only. No paid API calls.
- **4 GB VRAM is the ceiling**, and it is shared with the Windows desktop and
  the browser rendering our own captions. Model size and `compute_type` are
  config, decided by `docs/BENCHMARK.md` — do not hard-code them (D18, D20).
- Transcription runs in a **separate process** from audio capture (GIL).
- Append to `docs/DECISIONS.md` when you make a real decision. Update
  `docs/STATE.md` when you finish work.
- Log prompts and outcomes to `docs/journey/` — it is a graded deliverable.
