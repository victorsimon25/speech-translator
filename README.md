# Speech Translator

Real-time captions for online meetings. It captures **system audio** — whatever
your speakers are playing — transcribes it locally on the GPU, translates it, and
shows running captions in a browser window beside the meeting.

> **Status: fully built — all levels 0–7 complete.** p95 word age **5.9 s**
> measured end-to-end on the demo machine (GTX 1650 Ti). Queue depth ≤ 1 for
> 89 % of utterances. See [`docs/STATE.md`](docs/STATE.md) for the full record.

## Why system audio

The user needs to understand what *other people* say. Those voices arrive as
audio played out of the speakers, so the operating system's output is the correct
tap point — and it is the only approach that is agnostic to the meeting platform.
Zoom, Teams, Meet, Slack, Discord and a YouTube video all play through the OS, so
one code path covers all of them (D1).

The accepted cost: the captured stream is a single mixed waveform, so speaker
identity is gone before we ever see the audio (D9).

## Pipeline

```
AudioSource → Segmenter → Transcriber → SentenceSplitter → Translator → Publisher → UI
 (WASAPI       (Silero VAD  (Whisper,     (carry-over        (Google       (WebSocket)   (browser)
  loopback)     + max-len)   CUDA)         fragment)          Cloud)
                            └─ worker process ─┘
```

Every arrow is a queue. Transcription runs in its **own process** — a CPU-bound
Whisper call in a thread would stall audio capture through the GIL, which is a
correctness requirement rather than an optimisation (D15).

Two processes total: the worker holds the CUDA context and nothing else;
everything else — capture, VAD, segmentation, sentence splitting, translation,
web server — lives in one asyncio process. Translation is network-bound, so
putting it behind the GPU would idle the scarcest resource (D27).

## Stack

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.12 | — |
| Capture | PyAudioWPatch (WASAPI loopback) | maintained PyAudio fork with loopback + prebuilt wheels (D3) |
| Resampling | `soxr` streaming resampler | filter state carries across chunks; per-chunk resampling clicks |
| VAD | Silero, **vendored ONNX** + `onnxruntime` | the PyPI package imports torch unconditionally — a second CUDA runtime on a 4 GB card (D35) |
| ASR | `faster-whisper` (CTranslate2) on CUDA | local, free, and does language ID + punctuation in one pass (D6, D16) |
| Translation | Google Cloud Translation, **REST + API key** | 500 k chars/month free, no per-request ceiling (D7); REST avoids a machine-specific credential path (D37) |
| Server | FastAPI + uvicorn | one process serves the static UI and the WebSocket (D38) |
| UI | Vanilla HTML/CSS/JS | no npm, no bundler, no build step between a change and a demo (D38) |
| Deps | `uv` + committed `uv.lock` | one lockfile resolving Windows *and* Linux (D39) |

**No PyTorch anywhere.** See D35 — this is deliberate and load-bearing.

## Where it runs

**Target platform is Windows** — GTX 1650 Ti, 4 GB VRAM. Code is written on Linux
and tested on Windows with git as the bridge (D22), so the package must import
cleanly on Linux: Windows-only imports are lazy and platform deps carry
`sys_platform` markers.

Linux is descoped to the `AudioSource` interface only (D17). The demo machine is
the only machine that has to run this; there is no CPU fallback (D36).

## Specifications

### Audio contract — fixed everywhere

| | |
|---|---|
| Sample rate | 16 000 Hz |
| Channels | mono |
| Sample format | signed 16-bit little-endian |
| Frame size | **512 samples / 32 ms** — Silero's required size (D23) |

Downmix and resampling happen **inside** the AudioSource, so no downstream stage
contains a sample-rate branch (D24).

### Latency budget

```
word_age = silence_threshold + MT_roundtrip + max_utterance_ms × (1 + RTF)
```

`word_age` is how stale the first word of a caption is by the time it is read. It
is the metric the product is judged on — **RTF is a stability metric, not a
latency one** (D25). A config holding RTF at 0.4 can still put captions 12 s
behind.

| Config | Value | Derivation |
|---|---|---|
| `silence_threshold_ms` | 600 | latency budget |
| `max_utterance_ms` | 4000 | largest `D` meeting a 7 s word-age target at the RTF gate |
| `min_utterance_ms` | 300 | below this it is a VAD blip, not speech |
| `max_utterance_floor_ms` | 2000 | floor for adaptive shrink under load |
| `asr_queue_maxsize` | 4 | ≈16 s backlog ceiling |
| `model_size`, `compute_type` | **TBD** | decided by [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — never hard-coded |

### Performance gates — all three must pass

1. Sustained **RTF < 0.5** over 10 minutes (stability).
2. Peak VRAM leaves room for the Windows desktop **and** the browser rendering
   our own captions — the budget is not 4 GB (D18).
3. **p95 word age < 7 s**, measured end to end (D26).

### Hard constraints

- Free tiers only — no paid API calls.
- 4 GB VRAM is the ceiling, shared with the desktop and the browser.
- 500 000 translated characters/month, no roll-over. The counter is persisted
  across restarts, because the realistic way to burn it is a testing loop (D31).

## Quick start (Docker — no GPU required)

```bash
git clone <repo>
cd speech-translator
docker compose up --build
# open http://localhost:8000
```

`DEMO_MODE=1` is set in the Docker image by default. The server starts with
`FakeTranscribeWorker` — no GPU or audio hardware needed. Captions are
placeholder text; the full UI (subtitle overlay, Document PiP, language
selector) works normally (D72).

## Usage (Windows — full pipeline)

```bash
uv sync                          # identical versions on both machines
cp .env.example .env             # optional: add your email to GOOGLE_TRANSLATE_API_KEY
                                 # (repurposed as MyMemory de= param for higher quota, D68)

python -m speech_translator.doctor          # preflight: UTF-8, lockfile, no-torch,
                                            # CUDA, cuDNN DLLs, vendored VAD,
                                            # loopback device, translation
python -m speech_translator.doctor --no-net # skip the live translation call
python -m speech_translator.doctor --json   # machine-readable

# Capture tools (Windows-only; list_devices says so on Linux)
python -m speech_translator.tools.list_devices
python -m speech_translator.tools.record_loopback -t 600 -o meeting.wav
python -m speech_translator.tools.record_loopback --input-wav any.wav -o out.wav

# Segmentation inspection
python -m speech_translator.tools.dump_utterances --input-wav meeting.wav
python -m speech_translator.tools.dump_utterances --input-wav meeting.wav \
    --write-wav utts/          # one WAV per utterance — play them
python -m speech_translator.tools.dump_utterances --max-utterance-ms 2000 --json

# Run the full pipeline
python -m speech_translator                 # → http://localhost:8000
```

`record_loopback` writes 16 kHz mono int16 — the pipeline's own frame format, and
the input `docs/BENCHMARK.md` consumes (D47). `--input-wav` swaps a WAV in for
the device so the tool runs on the Linux dev box (D48).

`dump_utterances` is Level 2's inspection instrument. The table tells you the
state machine is self-consistent; `--write-wav` is what tells you the cuts landed
where a person would have put them, because you can listen to them.

Run `doctor` first after every `git pull` on the Windows machine. It exists so a
failure names itself instead of surfacing three layers away (D40). On Linux it
reports CUDA, cuDNN and loopback as **FAIL** — that is the correct output on a
machine that is not the target, and those failures do not set the exit code
(D44).

### Tests

```bash
uv run pytest
```

On the Linux dev laptop, ROS is on the global `PYTHONPATH` and pytest autoloads
a broken plugin from it; run `PYTHONPATH= uv run pytest` there. That is a
property of that machine, not of this project, so it is not worked around in
`pyproject.toml`.

## Repo layout

```
docs/
  PRD.md           what we're building and its Definition of Done
  PLAN.md          the build ladder — levels 0-8, each with acceptance criteria
  STATE.md         where we are right now  ← read first
  INTERFACES.md    module contracts; read only the stage you're touching
  DECISIONS.md     why things are the way they are (append-only, D1-D72)
  BENCHMARK.md     the Whisper measurement that picks model + compute_type
  journey/         prompts, LLM assessments, and what was learned — graded
speech_translator/ application code
  config.py        the D25 values
  doctor.py        preflight (D40)
  logging_setup.py UTF-8 forced at package import (D41, D42)
  windows_cuda.py  puts the wheel CUDA DLLs on the loader path (D43)
  server/
    app.py         FastAPI: GET /, GET /subtitle, WS /ws
    static/
      index.html + app.js + style.css   main UI (BroadcastChannel relay, PiP)
      subtitle.html + subtitle.css + subtitle.js   floating caption overlay
assets/            vendored silero_vad.onnx + its MIT license (D35)
Dockerfile         demo mode image — no GPU required
docker-compose.yml single-command startup
tests/
```

## Which doc to read

| You want to know | Read |
|---|---|
| What is done, in progress, blocked | `docs/STATE.md` |
| What to build next, and when it's finished | `docs/PLAN.md` |
| What a stage's inputs and outputs are | `docs/INTERFACES.md` |
| Why a choice was made | `docs/DECISIONS.md` |
| What the product must do | `docs/PRD.md` |

`DECISIONS.md` is append-only and entries are never rewritten — when a decision
is superseded, the new entry says so. A record of what was true at the time is
worth more than a tidy file (D16).
