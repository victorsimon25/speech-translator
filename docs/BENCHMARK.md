# Benchmark — Whisper on this CPU

**Run this first, before building anything.** Its result decides the
transcription quality ceiling and can invalidate the local-Whisper plan
entirely. Everything else in the project is downstream of this number.

## Why this gates the project

Transcription must run **faster than audio arrives**. If Whisper takes 3 seconds
to transcribe 2 seconds of audio, the result is not a 1-second delay — it is a
delay that grows by 1 second every 2 seconds. Ten minutes into a meeting you are
minutes behind and the queue is unbounded.

This is the **real-time factor**:

```
RTF = processing wall-clock time / duration of the audio processed
```

RTF < 1 means the design works. RTF > 1 means that model size is unusable on
this machine at any latency target. It is binary, not a matter of tuning.

## Target

**Largest model whose _sustained_ RTF stays below 0.5.**

Half, not 1.0, because the remaining budget must cover translation, the UI, the
browser, the meeting client itself, and thermal throttling. An app that is
exactly keeping up has no margin for the moment a demo needs it.

## Hardware under test

Intel i7-8650U — 4 cores / 8 threads, 1.9 GHz base, 4.2 GHz turbo, AVX2, no
AVX-512. **No NVIDIA GPU** (Intel UHD 620 only). 15 GiB RAM. Ubuntu 24.04,
Python 3.12.3.

Repeat on the Windows machine afterwards; results will differ.

## Method

1. Install `faster-whisper` (CTranslate2 backend, `int8` quantisation, CPU).
   `whisper.cpp` is a reasonable alternative given AVX2 — worth comparing if
   `faster-whisper` disappoints.
2. Prepare a **realistic** audio sample: 10+ minutes of natural speech in a
   non-English language, ideally meeting-like rather than studio-clean.
3. For each model size — `tiny`, `base`, `small`, `medium` — process the sample
   in chunks matching the planned utterance length (6–8 s), and record
   per-chunk processing time.
4. Record the **whole run**, not a warm-up slice.

## This must be a sustained test

A U-series laptop chip cannot hold 4.2 GHz across all cores for minutes. A cold
30-second benchmark will flatter the machine and then fail live, which is the
worst possible time to discover it.

- Run for **at least 10 minutes continuously** per model size.
- Report RTF for the **first minute** and the **last minute** separately. The
  difference is the throttling penalty, and it is the number that predicts demo
  behaviour.
- Note CPU temperature or clock speed over the run if convenient.

## Report

Write results into this file and update `STATE.md`.

| Model | RTF (first min) | RTF (last min) | Under 0.5 sustained? | Notes |
|---|---|---|---|---|
| tiny | | | | |
| base | | | | |
| small | | | | |
| medium | | | | |

Then state plainly:

- **Selected model size**, and the RTF that justifies it.
- **Throttling penalty** observed between first and last minute.
- Whether any headroom remains for the diarization stretch goal (D9). Expect
  the answer to be no.

## If nothing clears the bar

If even `tiny` cannot hold RTF < 0.5 sustained, local transcription is not
viable on this machine and the plan must change. Options, in order of
preference — raise them with the user rather than choosing unilaterally:

1. Run transcription on the Windows machine if its CPU is stronger.
2. Reconsider the free-tier-only constraint for a cloud speech API (D8 —
   the user has already declined this once).
3. Reduce scope: translate short prepared clips rather than continuous meeting
   audio, and be explicit about the limitation.

## Journey doc

This benchmark is strong material. Capture the raw numbers, the throttling
curve, and the reasoning that turned them into a model choice. "I measured
rather than assumed" is exactly the kind of decision the assignment is asking
you to evidence — and a negative result, documented honestly, is worth more than
an untested assumption that happened to hold.
