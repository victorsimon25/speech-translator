# State

_Last updated: 2026-08-18 (design session)_

## Done
- Requirements and architecture agreed (see `DECISIONS.md`).
- Hardware surveyed and audio capture feasibility **verified on Linux**
  (PipeWire monitor ports present on sink node 47).
- Windows capture path identified: `PyAudioWPatch` (WASAPI loopback,
  prebuilt wheels for Python 3.12).
- Translation provider chosen: Google Cloud Translation, free tier.
- Module boundaries defined (`INTERFACES.md`).
- Repo initialised, docs written.

## Next (in order)
1. **Run the Whisper benchmark** — `BENCHMARK.md`. This is the first task of the
   next session and it gates everything downstream. Do not build the pipeline
   before this number exists.
2. Implement `AudioSource` (Linux first) + a `WavFileSource` for deterministic
   tests.
3. Implement `Segmenter` (VAD + max-length cap).
4. Wire `Transcriber` using the model size the benchmark selected.
5. Add `Translator` against Google Cloud Translation.
6. Build the caption UI.

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
