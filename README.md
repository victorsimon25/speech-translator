# Speech Translator

Real-time captions for online meetings. It captures **system audio** — whatever
your speakers are playing — transcribes it locally on the GPU, translates it, and
shows running captions in a browser window beside the meeting.

> **Status: design complete, no application code yet.** 41 recorded decisions,
> contracts frozen, build ladder defined. See [`docs/STATE.md`](docs/STATE.md)
> for where things stand and [`docs/PLAN.md`](docs/PLAN.md) for what gets built
> next. The usage section below describes the **target** interface.

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

## Usage *(target interface — not yet built)*

```bash
uv sync                                   # identical versions on both machines
cp .env.example .env                      # add GOOGLE_TRANSLATE_API_KEY

python -m speech_translator.doctor         # preflight: CUDA, cuDNN, model cache,
                                           # loopback device, credentials
python -m speech_translator.tools.list_devices
python -m speech_translator.tools.record_loopback sample.wav --seconds 600

python -m speech_translator                # then open http://localhost:8000
```

Run `doctor` first after every `git pull` on the Windows machine. It exists so a
failure names itself instead of surfacing three layers away (D40).

## Repo layout

```
docs/
  PRD.md           what we're building and its Definition of Done
  PLAN.md          the build ladder — levels 0-8, each with acceptance criteria
  STATE.md         where we are right now  ← read first
  INTERFACES.md    module contracts; read only the stage you're touching
  DECISIONS.md     why things are the way they are (append-only, D1-D41)
  BENCHMARK.md     the Whisper measurement that picks model + compute_type
  journey/         prompts, LLM assessments, and what was learned — graded
speech_translator/ application code (not yet written)
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
