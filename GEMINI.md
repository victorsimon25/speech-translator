# Speech Translator

Real-time speech translator for online meetings. Captures **system audio**
(what the speakers are playing), transcribes it locally, translates it, and
shows captions in the browser.

> This file is the Gemini / Antigravity counterpart of `CLAUDE.md`. Both carry
> the same project rules — they must not drift apart. If you change a rule here,
> change it there too, and say so in the commit.
>
> If your setup loads `AGENTS.md` rather than `GEMINI.md`, copy this file to that
> name; nothing in it is agent-specific except the section at the bottom.

## You are most likely on the Windows demo machine

This repo is written on a Linux laptop and tested on a Windows machine, with git
as the bridge (D22). Almost all the code exists already; what the Windows machine
owes is **measurements** — four levels' worth, none of which can be taken
anywhere else.

**Read [`docs/WINDOWS.md`](docs/WINDOWS.md) first.** It is the runbook for that
session: closing Levels 0-2, running Level 3's benchmark, and the list of docs to
update afterwards.

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
1. `docs/WINDOWS.md` — **if you are on the Windows machine, this is the task.**
2. `docs/STATE.md` — what's done, in progress, next, blocked. **Read first.**
3. `docs/PLAN.md` — what gets built next and what "done" means for it.
   Work one level at a time.
4. `docs/INTERFACES.md` — module contracts. Read the stage you're touching;
   you should not need to read another stage's source.
5. `docs/DECISIONS.md` — why things are the way they are. Check before
   proposing a change.

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

## The rule that matters most on this machine

**Everything here is a measurement.** Report what the machine says, including
when it is bad news. Never estimate a number, never fill a results cell you did
not observe, never tick an acceptance box you did not earn. A documented negative
result is worth more than an assumption that happened to hold — that is
explicitly what the assignment is asking to see evidenced.

## Working notes for Antigravity

- **Ask before starting anything long.** The benchmark matrix is ~75 minutes of
  continuous GPU work plus ~6-7 GB of model downloads. Confirm with the user
  first, and tell them what you need from them (play audio, open a browser,
  listen to a file).
- **Do not kill a long run to check on it.** The benchmark writes
  `results.json`, `results.md`, `chunks-*.jsonl` and `gpu.jsonl` incrementally
  into `var/benchmark/<timestamp>/`. Read those files while it runs instead.
- **Do not parallelise the benchmark across agents.** Every configuration is
  competing for one 4 GB card; two at once measures neither. One agent, one run,
  in order.
- **Two acceptance criteria need human ears, not a tool.** "The recording plays
  back cleanly" and "boundaries land at real pauses" are listening tests. Stop
  and ask the user to listen; do not substitute a waveform check and call it
  done.
- **The shell is PowerShell.** Commands in `docs/WINDOWS.md` are written for it.
  Use `uv run python -m ...` so the project venv is used without activating it.
- **Terminal output is the evidence.** Keep it — the journey doc wants raw
  numbers and the actual error text, especially for anything that went wrong.
- If you produce a plan or task-list artifact, it is scaffolding, not a
  deliverable. The deliverables are the filled tables in `docs/BENCHMARK.md`, the
  updated `docs/STATE.md` / `docs/PLAN.md`, and `docs/journey/08-*.md`.
