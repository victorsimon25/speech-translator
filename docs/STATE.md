# State

_Last updated: 2026-08-18 (session 2 — platform pivot)_

## Done
- Requirements and architecture agreed (see `DECISIONS.md`).
- Translation provider chosen: Google Cloud Translation, free tier.
- Module boundaries defined (`INTERFACES.md`).
- **Platform pivot to Windows + CUDA** (GTX 1650 Ti, 4 GB). Docs retargeted;
  reasoning in `DECISIONS.md` D16–D24.
  - Linux descoped to the `AudioSource` interface only (D17). The PipeWire
    monitor path was verified in D2 and stays on record, but is not built.
  - Dev on Linux, test on Windows, git as the bridge (D22).
  - VAD settled: Silero, 512-sample / 32 ms frames (D23).
- Repo initialised, docs written.

## Next (in order)
1. `AudioSource` — `WasapiLoopbackSource` (Windows, the live source) and
   `WavFileSource` (tests + Linux smoke path). Includes downmix/resample to
   16 kHz mono int16 inside the source (D24).
2. `Segmenter` — Silero VAD, silence threshold **or** max-length cap (D12).
3. **Run the Whisper benchmark** — `BENCHMARK.md`, on the Windows machine.
   Must happen before the Transcriber is wired; it decides model size and
   `compute_type`, and whether D9 is reachable.
4. Wire `Transcriber` using whatever the benchmark selected.
5. Add `Translator` against Google Cloud Translation.
6. Build the caption UI.

**Note on ordering:** the benchmark used to be step 1 and marked as gating
everything. It moved because CUDA answered the existential question it was
asked to answer — see D20. Steps 1 and 2 are model-independent, so building them
first costs nothing and gives the benchmark a real capture path to source its
sample audio from.

## In progress
Nothing. No application code exists yet.

## Blocked
- **Three truncated lines in the assignment brief** are still unknown — see
  `PRD.md`. The scope line gates whether incoming-only is permitted.
- **Deadline unknown.** The journey doc is due 48h before the interview, so the
  real deadline is earlier than the interview date.

## Open questions
- Which reading of the silence/punctuation bonus is intended (correct
  punctuation in output vs. visibly showing pauses in the UI)? See `PRD.md`.
- Desktop shell (Electron/Tauri) around the browser UI — deferred, revisit once
  the core pipeline works.
- Whether `int8_float16` beats `float16` on a GPU with no tensor cores. The
  benchmark will say; do not assume the RTX answer transfers (D18).
