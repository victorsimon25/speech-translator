# Speech Translator

Real-time speech translator for online meetings. Captures **system audio**
(what the speakers are playing), transcribes it locally, translates it, and
shows captions in the browser.

## Stack
Python 3.12 backend + browser UI over WebSocket. Local Whisper (CPU) for
transcription, Google Cloud Translation (free tier) for translation.

## Read before working
1. `docs/STATE.md` — what's done, in progress, next, blocked. **Read first.**
2. `docs/INTERFACES.md` — module contracts. Read the stage you're touching;
   you should not need to read another stage's source.
3. `docs/DECISIONS.md` — why things are the way they are. Check before
   proposing a change.

## Hard rules
- Free tiers only. No paid API calls.
- No CUDA on the dev machine — transcription is CPU-bound. Check
  `docs/BENCHMARK.md` before assuming a model size is viable.
- Transcription runs in a **separate process** from audio capture (GIL).
- Append to `docs/DECISIONS.md` when you make a real decision. Update
  `docs/STATE.md` when you finish work.
- Log prompts and outcomes to `docs/journey/` — it is a graded deliverable.
