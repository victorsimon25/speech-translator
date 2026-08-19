# Speech Translator

Real-time speech translator for online meetings. Captures **system audio**
(what the speakers are playing), transcribes it locally, translates it, and
shows captions in the browser.

## Stack
Python 3.12. Two processes: a `faster-whisper` CUDA worker, and everything else
(capture, VAD, segmentation, translation, web server) in one asyncio process (D27).

| Layer | Choice |
|---|---|
| Capture | PyAudioWPatch (WASAPI loopback), lazy import |
| Resampling | `soxr` **streaming** resampler — state carries across chunks |
| VAD | Silero, **vendored ONNX** + `onnxruntime` |
| ASR | `faster-whisper` (CTranslate2) on CUDA |
| Translation | Google Cloud Translation, **REST + API key** (not the client lib) |
| Server / UI | FastAPI + uvicorn; vanilla HTML/CSS/JS, no npm, no build step |
| Deps | `uv` + committed `uv.lock` |

**No PyTorch anywhere** — see the hard rules.

## Where this runs
- **Target platform: Windows.** GTX 1650 Ti, 4 GB VRAM. Linux is descoped to the
  `AudioSource` interface only (D17).
- **Written on Linux, tested on Windows**, with git as the bridge (D22). Code
  must import cleanly on Linux: Windows-only imports are lazy, and
  platform-specific deps carry environment markers.

## Read before working
1. `docs/STATE.md` — what's done, in progress, next, blocked. **Read first.**
2. `docs/PLAN.md` — what gets built next and what "done" means for it.
   Work one level at a time.
3. `docs/INTERFACES.md` — module contracts. Read the stage you're touching;
   you should not need to read another stage's source.
4. `docs/DECISIONS.md` — why things are the way they are. Check before
   proposing a change.

**If you are running on the Windows demo machine**, `docs/WINDOWS.md` is the
runbook: closing Levels 0-2 and executing Level 3's benchmark, with the docs to
update afterwards.

## Hard rules
- Free tiers only. No paid API calls.
- **4 GB VRAM is the ceiling**, and it is shared with the Windows desktop and
  the browser rendering our own captions. Model size and `compute_type` are
  config, decided by `docs/BENCHMARK.md` — do not hard-code them (D18, D20).
- Transcription runs in a **separate process** from audio capture (GIL).
- **Never add `torch` or `torchaudio`.** The `silero-vad` package imports torch
  unconditionally; a second CUDA runtime alongside CTranslate2 on a 4 GB card is
  a real failure, not a theoretical one. Use the vendored ONNX (D35).
- Dependencies go through `uv` and the committed `uv.lock` — not `pip install`,
  not `requirements.txt` (D39).
- The config values in D25 are **derived, not defaults**. Read D25 before
  touching `max_utterance_ms`: it is the dominant latency knob, not RTF.
- **Force UTF-8** on every output stream and log file. The Windows console is
  cp1252 and this app's entire output is non-ASCII (D41).
- Don't start a `docs/PLAN.md` level before the previous level's acceptance
  criteria pass. Update its status column when one lands.
- Append to `docs/DECISIONS.md` when you make a real decision. Update
  `docs/STATE.md` when you finish work.
- Log prompts and outcomes to `docs/journey/` — it is a graded deliverable.
