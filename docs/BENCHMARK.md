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

## Three gates. All must pass.

**1. Sustained RTF < 0.5.**

Half, not 1.0, because the remaining budget must cover translation, the UI, the
browser, the meeting client itself, and thermal throttling. An app that is
exactly keeping up has no margin for the moment a demo needs it.

**This is a stability gate. It is not a latency gate** — see gate 3.

**2. Peak VRAM leaves room for the desktop and the browser.**

This is the new binding constraint (D18). 4 GB total, and the Windows desktop
compositor plus the browser rendering our own captions are already drawing from
it. Record VRAM already in use *at idle* during the hardware survey and subtract
it — the budget is not 4 GB.

The failure mode here is abrupt: not "gradually falls behind" but **CUDA OOM**,
possibly not until a long utterance produces an unusually large activation.

**3. p95 word age < 7 s**, at `max_utterance_ms = 4000`.

Added in D25/D26, because gates 1 and 2 together were passable by a
configuration that is unusable. Word age is how stale the *first word* of a
caption is by the time the user reads it:

```
word_age = silence_threshold + MT_roundtrip + D × (1 + RTF)
```

At the old defaults (`D` = 8 s, silence 700 ms, RTF 0.4, MT 300 ms) that is
**12.2 seconds** — and it passes gates 1 and 2 comfortably. Twelve seconds
behind is a subtitled recording, not a meeting anyone can take part in.

**This gate is evaluated twice, and the two are different claims (D51).**

| | Level 3 — *predicted* | Level 7 — *measured* |
|---|---|---|
| Where the numbers come from | measured RTF from this benchmark, plus a **measured** MT round trip (time the real API call — `doctor` already makes one), plus the config's `silence_threshold_ms` and `max_utterance_ms` | a live 10-minute session, timestamped end to end through the JSONL log (D34) |
| What it can see | the dominant terms | queue wait, scheduling, the browser competing for the GPU |
| What it is for | **choosing the model** — this is the column you select on | **evidencing the DoD** |

Level 3 cannot measure this end to end, because at Level 3 there is no end: the
Transcriber is not wired, the Translator does not exist, and there is no UI.
Predicting it here is not a shortcut, it is the only thing available — and it is
sufficient for its actual job, which is ranking seven configurations against each
other. What it must **not** be is quietly reported as a measurement. Label the
column *predicted* in the results table.

The one term you can and should measure rather than assume at Level 3 is
`MT_roundtrip`: it is a network call to a live API, its latency is real, and
plugging in a guessed 300 ms is exactly the kind of substitution D25 exists to
prevent. Time ten of them and take the p95.

### Why this changes which model wins

RTF pays **twice** — it multiplies `D` in the formula above. So the selection
rule is **the fastest model that is accurate enough, not the largest that
fits**. A model that fits in VRAM and clears RTF 0.49 is a worse choice than one
at RTF 0.25, even though the table's first two columns would call them both a
pass. Fill in the word-age column and select on it.

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
   Windows trap is that the required cuBLAS and cuDNN DLLs are not bundled.
   Take them from the **`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip wheels**
   pinned in `uv.lock` (D39, D41) rather than hand-copying DLLs, which is not
   reproducible on the next machine. Verify with
   `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`
   returning `1` **before** benchmarking anything — `python -m speech_translator.doctor`
   checks this and the rest of the register (D40). Write down whatever worked.
2. Sample audio: 10+ minutes of natural speech in a non-English language,
   meeting-like rather than studio-clean. Capture it with
   `python -m speech_translator.tools.record_loopback` while playing a
   Spanish-language interview or podcast — the capture stage produces its own
   benchmark input, and that exercises the real path rather than a curated file.
   Keep the recipe here; keep the audio out of git.
3. Process the sample in chunks matching the planned utterance length. **Level 2
   is built, so the chunks come from the real `Segmenter` (D57)** — the actual
   utterance-length distribution, with pre-roll prepended and closing silence
   trimmed (D55), which is byte-for-byte the audio the app will hand Whisper.
   Record per-chunk processing time.

   Chunk length is not a detail: per-call overhead is amortised over less audio,
   so RTF at 4 s can be meaningfully worse than at 8 s, and real utterances close
   on *silence* far more often than on max-length, so the true distribution skews
   shorter than 4 s. A fixed-4 s benchmark is **optimistic, not neutral** — which
   is the cost D52 was prepared to pay and D57 no longer has to. The harness
   records the distribution it actually cut (count, p50, p95, max, and the
   silence/max-length split) into `results.json`, so this is measured rather than
   assumed.

   The audio is cut **once, up front, and reused for all seven configurations**,
   so an RTF difference between two rows is a difference in the model and not in
   what was measured.

   **D52's bracket survives as a check on the selected row**: re-run the winner at
   fixed 4 s and fixed 1.5 s (`--chunking fixed --chunk-ms 1500`) and report all
   three. If a bracket fails a gate the headline run passes, that is the finding.
4. Benchmark **the configuration the app will actually use**, or the number is
   fiction:
   - `beam_size=1` — greedy, as the real-time path will be
   - `condition_on_previous_text=False` — prevents hallucination loops on
     chunked audio
   - `vad_filter=False` — our Segmenter owns VAD (D23)
   - `language` pinned — the app locks language after a short detection window
     (D32), so re-running language identification every chunk would overstate
     the cost
5. Exclude model load time from RTF; record it separately. Mark the first chunk
   as warm-up and report it separately too.
6. Record the **whole run**, not a warm-up slice.

## This must be a sustained test

A laptop GPU cannot hold boost clocks indefinitely any more than the U-series CPU
could. A cold 30-second benchmark will flatter the machine and then fail live,
which is the worst possible time to discover it.

- Run for **at least 10 minutes of wall clock** per configuration, looping the
  recording as needed (D59). Ten minutes of *audio* is not the same test: at
  RTF 0.3 that is a three-minute run, and a GPU that never gets hot cannot report
  a throttling penalty. Seven configurations x 10 wall-clock minutes is the ~70
  minutes this level is budgeted at.
- Report RTF for the **first minute** and the **last minute** separately, split on
  wall clock. The difference is the throttling penalty, and it is the number that
  predicts demo behaviour. The warm-up chunk is excluded from the first-minute
  figure, which it would otherwise dominate.
- Log GPU state across the run:

  ```
  nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv -l 1
  ```

- **Run it with a browser open**, because the demo has one and it is competing
  for the same 4 GB.

## The harness

Do not drive this from a REPL (D51). `python -m speech_translator.tools.benchmark`
runs the whole method:

```
python -m speech_translator.tools.benchmark --input-wav <the 10-minute capture>
```

- Defaults to the seven configurations below, 10 wall-clock minutes each,
  Segmenter chunking, `--language es`.
- Refuses to time anything until `ctranslate2.get_cuda_device_count()` returns 1
  (D36), and calls `add_cuda_dll_directories()` before importing it (D41, D43).
- Measures `MT_roundtrip` from ten live calls and takes the p95. **With no API
  key it records the term as missing and gate 3 reports `unknown`** — it never
  substitutes the guessed 300 ms (D51).
- Writes `results.json`, `results.md` (the table below, ready to transcribe),
  `chunks-*.jsonl` and `gpu.jsonl` into `var/benchmark/<timestamp>/`,
  **incrementally** — a configuration that OOMs is a failed row and the matrix
  continues to the next one.
- `--dry-run` exercises the whole harness with a stub engine, no weights and no
  GPU (D58). `--render results.json` re-renders the table without running.

It **reports; it does not choose.** Pre-download the weights before the timed
run, or the first configuration's `load_s` is mostly a download.

## Report

Write results into this file and update `STATE.md`. The table below is what
`results.md` emits — transcribe it rather than retyping it.

| Model | compute_type | RTF (first min) | RTF (last min) | Peak VRAM | p95 word age *(predicted)* | Under 0.5 sustained? | Word age < 7 s? | Fits alongside browser? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| small | float16 | | | | | | | | |
| small | int8_float16 | | | | | | | | |
| medium | float16 | | | | | | | | |
| medium | int8_float16 | | | | | | | | |
| large-v3-turbo | float16 | | | | | | | | |
| large-v3-turbo | int8_float16 | | | | | | | | |
| large-v3 | int8_float16 | | | | | | | | |

Also record, for the selected configuration, a **subjective accuracy note** in
the demo's source language. The selection rule is "fastest that is accurate
enough", and "enough" has to be judged by listening, not by the RTF column.

Then state plainly:

- **Selected model size and `compute_type`**, with the RTF, the peak VRAM *and*
  the p95 word age that justify them.
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
   the cost of more mid-sentence cuts (the D12 tradeoff, now partly paid down by
   sentence carry-over, D29). Floor is 2000 ms (D28). Note that this *helps*
   gate 3 while hurting translation quality.

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
