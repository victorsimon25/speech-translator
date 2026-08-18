# Benchmark — Whisper on the Windows GPU

**Status: not yet run.** It is no longer a blocking gate on all work, but it
still runs **before the Transcriber is wired**. See D20 for why that changed.

Its result decides two config values — model size and `compute_type` — and
whether the diarization stretch goal (D9) is reachable.

## What this measures, and why it still matters

Transcription must run **faster than audio arrives**. If Whisper takes 3 seconds
to transcribe 2 seconds of audio, the result is not a 1-second delay — it is a
delay that grows by 1 second every 2 seconds. Ten minutes into a meeting you are
minutes behind and the queue is unbounded.

This is the **real-time factor**:

```
RTF = processing wall-clock time / duration of the audio processed
```

On CPU this was an open question that could have invalidated the whole design.
On CUDA it is near-certain that *something* clears the bar, so the benchmark's
job has shifted from "is this possible" to "how much can we afford, and does it
fit in memory" (D20).

## Two gates. Both must pass.

**1. Sustained RTF < 0.5.**

Half, not 1.0, because the remaining budget must cover translation, the UI, the
browser, the meeting client itself, and thermal throttling. An app that is
exactly keeping up has no margin for the moment a demo needs it.

**2. Peak VRAM leaves room for the desktop and the browser.**

This is the new binding constraint (D18). 4 GB total, and the Windows desktop
compositor plus the browser rendering our own captions are already drawing from
it. Record VRAM already in use *at idle* during the hardware survey and subtract
it — the budget is not 4 GB.

The failure mode here is abrupt: not "gradually falls behind" but **CUDA OOM**,
possibly not until a long utterance produces an unusually large activation.

## Hardware under test

**To be filled in by the survey, not estimated.** Session 1 established the
practice of inspecting the machine rather than guessing at it, and that is worth
keeping.

Record, on the Windows machine:

```
nvidia-smi                     # driver version, CUDA version, total VRAM, VRAM in use at idle
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -VV                     # confirm 3.12
```

Plus CPU model, core count, and RAM.

| | |
|---|---|
| GPU | GeForce GTX 1650 Ti, 4 GB GDDR6 (TU117, compute capability 7.5, **no tensor cores**) |
| Driver / CUDA | _to fill_ |
| VRAM in use at idle | _to fill — this comes off the budget_ |
| CPU / RAM | _to fill_ |
| OS / Python | Windows, Python 3.12 |

## Model shortlist

Multilingual only. **Distil-Whisper is excluded** — it is English-only, and the
product premise is transcribing a language the user does not speak (D19).

| Model | Params | ~fp16 weights | Why it is on the list |
|---|---|---|---|
| `small` | 244M | ~0.5 GB | safe floor, near-certain pass |
| `medium` | 769M | ~1.5 GB | likely comfortable |
| `large-v3-turbo` | 809M | ~1.6 GB | multilingual, pruned decoder — likely the sweet spot |
| `large-v3` | 1550M | ~3.1 GB fp16 / ~1.6 GB int8 | best accuracy; fp16 probably will not fit |

Test each at **`float16`** and **`int8_float16`**. Do not assume int8 wins: this
part has no tensor cores (D18), so the usual RTX intuition does not transfer.

Weight sizes above are the model only. Activations, the CUDA context, and
cuDNN workspace are extra — which is why peak VRAM is measured rather than
calculated.

## Method

1. Environment: `faster-whisper` (CTranslate2 backend) on CUDA. The known
   Windows trap is that the required cuBLAS and cuDNN DLLs are not bundled —
   resolve that and verify with
   `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`
   returning `1` **before** benchmarking anything. Write down whatever worked.
2. Sample audio: 10+ minutes of natural speech in a non-English language,
   meeting-like rather than studio-clean. Capture it with
   `python -m speech_translator.tools.record_loopback` while playing a
   Spanish-language interview or podcast — the capture stage produces its own
   benchmark input, and that exercises the real path rather than a curated file.
   Keep the recipe here; keep the audio out of git.
3. Process the sample in chunks matching the planned utterance length (6–8 s,
   per D12), recording per-chunk processing time.
4. Benchmark **the configuration the app will actually use**, or the number is
   fiction:
   - `beam_size=1` — greedy, as the real-time path will be
   - `condition_on_previous_text=False` — prevents hallucination loops on
     chunked audio
   - `vad_filter=False` — our Segmenter owns VAD (D23)
   - `language` pinned — the app locks language after the first utterance, so
     re-running language identification every chunk would overstate the cost
5. Exclude model load time from RTF; record it separately. Mark the first chunk
   as warm-up and report it separately too.
6. Record the **whole run**, not a warm-up slice.

## This must be a sustained test

A laptop GPU cannot hold boost clocks indefinitely any more than the U-series CPU
could. A cold 30-second benchmark will flatter the machine and then fail live,
which is the worst possible time to discover it.

- Run for **at least 10 minutes continuously** per configuration.
- Report RTF for the **first minute** and the **last minute** separately. The
  difference is the throttling penalty, and it is the number that predicts demo
  behaviour.
- Log GPU state across the run:

  ```
  nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv -l 1
  ```

- **Run it with a browser open**, because the demo has one and it is competing
  for the same 4 GB.

## Report

Write results into this file and update `STATE.md`.

| Model | compute_type | RTF (first min) | RTF (last min) | Peak VRAM | Under 0.5 sustained? | Fits alongside browser? | Notes |
|---|---|---|---|---|---|---|---|
| small | float16 | | | | | | |
| small | int8_float16 | | | | | | |
| medium | float16 | | | | | | |
| medium | int8_float16 | | | | | | |
| large-v3-turbo | float16 | | | | | | |
| large-v3-turbo | int8_float16 | | | | | | |
| large-v3 | int8_float16 | | | | | | |

Then state plainly:

- **Selected model size and `compute_type`**, with the RTF *and* the peak VRAM
  that justify them.
- **Throttling penalty** observed between first and last minute.
- Whether `int8_float16` actually beat `float16` on a part with no tensor cores.
- Whether any headroom remains for the diarization stretch goal (D9). Expect
  the answer to still be no — a diarization model wants VRAM the caption UI is
  already competing for.

## If nothing clears the bar

The realistic failure on CUDA is **out of memory**, not slowness. In order:

1. Drop to `int8_float16` — roughly halves the weight footprint.
2. Drop a model size.
3. Reduce `max_utterance_ms` — shorter utterances mean smaller activations, at
   the cost of more mid-sentence cuts (the D12 tradeoff).

If instead RTF is the problem even at `small`, that is a surprising result worth
investigating before working around: check that CUDA is genuinely being used and
the run has not silently fallen back to CPU. `get_cuda_device_count()` returning
`1` at startup is the guard against exactly that.

## Journey doc

This benchmark is strong material. Capture the raw numbers, the throttling curve,
and the reasoning that turned them into a model choice. "I measured rather than
assumed" is exactly the kind of decision the assignment is asking you to evidence
— and a negative result, documented honestly, is worth more than an untested
assumption that happened to hold.

Screenshots worth taking: `nvidia-smi` mid-run, the filled table, and the
first-minute/last-minute throttling curve.
