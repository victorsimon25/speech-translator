# State

_Last updated: 2026-08-18 (session 3 — system design requirements)_

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
- **System design requirements closed** (D25–D34). The project had a stability
  requirement but no latency requirement; it now has both, with the config
  values derived rather than guessed.
  - **Latency budget**: `word_age = silence + MT + D × (1 + RTF)`, target p95
    < 7 s. This is now a third benchmark gate (D26).
  - **Config is derived, not placeholder**: `silence_threshold_ms = 600`,
    `max_utterance_ms = 4000` (was 6000–8000), `min_utterance_ms = 300`,
    `max_utterance_floor_ms = 2000`, `asr_queue_maxsize = 4`.
  - Two processes exactly; translation stays with the server, not the GPU (D27).
  - Backpressure: adaptive shrink → drop oldest, visible in the UI (D28).
  - Caption unit is the **sentence**, with fragment carry-over and a flush
    rule (D29).
  - Hallucination guard, persisted character budget, windowed LID (D30–D32).
  - One JSONL timing log per session — the journey doc's evidence base (D34).
- Repo initialised, docs written.

## Next (in order)
1. `AudioSource` — `WasapiLoopbackSource` (Windows, the live source) and
   `WavFileSource` (tests + Linux smoke path). Includes downmix/resample to
   16 kHz mono int16 inside the source (D24). Plus `tools/list_devices` and
   `tools/record_loopback` — the latter produces the benchmark's sample audio,
   so it is on the critical path, not a side quest.
2. `Segmenter` — Silero VAD, silence threshold **or** max-length cap (D12), at
   the D25 values. Must accept a runtime override of the effective
   `max_utterance_ms` so the backpressure controller (D28) can shrink it.
   Plus `tools/dump_utterances`.
3. **Run the Whisper benchmark** — `BENCHMARK.md`, on the Windows machine.
   Must happen before the Transcriber is wired. Three gates now, and the
   selection rule is *fastest that is accurate enough*, not largest that fits.
4. Wire `Transcriber` (worker process + ready handshake, D27) using whatever the
   benchmark selected.
5. `SentenceSplitter`, then `Translator` against Google Cloud Translation with
   the persisted budget counter and LRU cache (D31).
6. Build the caption UI, including the six defined states (PRD).

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
  The design covers both cheaply — `preceding_silence_ms` reaches the UI on the
  first sentence of every utterance — so this is no longer urgent.
- Desktop shell (Electron/Tauri) around the browser UI — deferred, revisit once
  the core pipeline works.
- Whether `int8_float16` beats `float16` on a GPU with no tensor cores. The
  benchmark will say; do not assume the RTX answer transfers (D18).
- Sentence-boundary splitting for languages without Latin punctuation
  conventions (e.g. Thai, which does not space or full-stop the same way). Not a
  problem for the likely demo languages; note it before claiming generality.
