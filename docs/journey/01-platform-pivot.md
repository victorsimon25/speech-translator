# Journey 01 — Platform pivot to Windows + CUDA (2026-08-18)

Session with Claude (Opus 5) in Claude Code, in plan mode. No production code
yet; this session retargeted the design and planned the first two pipeline
stages.

## The prompt that started it

> Read docs/STATE.md and docs/BENCHMARK.md, and the testing is oriented over
> linux but i have a seperate windows system with a nvidia gpu so i dont wana
> build the project over linux

Followed, after the model had asked what happens to the Linux path and which GPU:

> i dont wana run the benchmark now and ill be working on the code in this
> system but for testing alone ill be switching to windows using git

## What the LLM got right

**It identified the load-bearing assumption rather than treating this as a
platform swap.** Its first observation was that "no NVIDIA GPU" from session 1
is not one fact among many — it is the fact that D9 (diarization descoped), D11
(final-only captions), the "main risk" framing in D6, and the entire existence of
`BENCHMARK.md` were derived from. So the pivot required re-examining each of
those, not just changing an OS name.

**It refused to declare the risk gone.** The tempting conclusion is "GPU,
therefore fast, therefore the descoped features come back." What it actually said
is that the risk *moves* rather than disappears:

| | Before | After |
|---|---|---|
| Binding constraint | CPU throughput | VRAM — 4 GB, shared with the desktop and the browser showing our own captions |
| Failure mode | Falls progressively behind | CUDA OOM, possibly not until a long utterance |
| Throttling | 4 U-series cores | Laptop GPU power/thermal limits — still real |

The point that the caption UI's own browser competes for the same 4 GB was one I
had not considered, and it changed how the benchmark must be run: with a browser
open, because the demo has one.

**It asked for the GPU model instead of assuming a tier.** The answer — GTX
1650 Ti — matters more than "has CUDA" would suggest. It noted the part has no
tensor cores, so the usual "int8 is faster than fp16" rule from RTX cards is not
guaranteed here, and put that in the benchmark as something to measure rather
than as a claim.

**It caught the distil-Whisper trap before I hit it.** Distil-Whisper is the
standard answer to "make Whisper faster," and the numbers would have looked
great. It is also **English-only** — which is fatal for an app whose entire
premise is transcribing a language the user does not speak. That is now D19,
recorded specifically because the mistake is attractive rather than obvious.

**It overruled a doc explicitly instead of quietly ignoring it.** `STATE.md`
said in bold "do not build the pipeline before this number exists." I asked to
skip the benchmark for now. Rather than just going along with it, it explained
why the rule could be relaxed — the benchmark was a gate when the question was
*existential* (can CPU Whisper keep up at all?), and CUDA answers that; what
remains is which model, which is a config value, not an architectural fork —
and then wrote that reasoning into `DECISIONS.md` as D20 so the next session
sees a deliberate decision rather than an ignored instruction.

That distinction — demote with reasons vs. silently skip — is the part of this
session I would most want to keep.

## Where I steered it

My first answer implied Windows would be the whole environment, and the plan it
drafted put all work on the Windows machine. That was not what I meant: I write
code on the Linux laptop and only *test* on Windows, over git. Correcting that
changed the plan materially — it added the requirement that the package must
import cleanly on Linux with no Windows dependencies installed (lazy imports,
environment markers on `PyAudioWPatch`), and it kept `WavFileSource` as a local
smoke path so I am not doing push-pull-run for every typo. That is now D22.

Worth noting for my own process: the model asked structured questions before
planning rather than after, which is what surfaced the misunderstanding at a
point where fixing it cost one message instead of a rewrite.

## Decisions this session produced

Full reasoning in `DECISIONS.md` D16–D24. The load-bearing ones:

- **D18** — VRAM, not CPU throughput, is now the binding constraint, and the
  budget is not the full 4 GB.
- **D19** — Distil-Whisper excluded: English-only, incompatible with the product.
- **D20** — Benchmark demoted from blocking gate to model-selection task, with
  the reasoning recorded rather than the doc quietly overridden.
- **D21** — D9 and D11 stay closed until there are numbers. A 4 GB laptop part
  is not obviously enough to reopen them, and reopening on optimism is the exact
  assume-don't-measure error the benchmark exists to prevent.
- **D23** — VAD settled as Silero, which fixes the frame size at 512 samples /
  32 ms. Chosen over WebRTC VAD because the input is a whole desktop's audio —
  music, notifications, video stings — and every false positive is a wasted GPU
  call plus a hallucinated caption.

## What I learned

Session 1's lesson was to measure rather than assume. This session's is the
follow-on: when the thing you measured *changes*, the conclusions you drew from
it do not update themselves. Getting a GPU felt like unblocking the project, and
the honest reading is that it moved the bottleneck from one resource to another
and made the failure mode less gradual and easier to miss.

The second lesson is about the docs themselves. Because session 1 wrote down
*why* each decision was made, this session could tell which ones actually
depended on the CPU constraint and which merely coincided with it. A list of
decisions without reasons would have left me guessing at exactly the moment the
guessing was expensive.

## Next

Build `AudioSource` (WASAPI loopback + WAV) and the `Segmenter`. Then run the
benchmark on Windows, using audio captured through the capture stage itself.

---

**Screenshots to capture next session:** `list_devices` showing the WASAPI
loopback device, a captured 16 kHz mono WAV opened in a waveform viewer,
and `dump_utterances` output showing utterance boundaries against live meeting
audio.
