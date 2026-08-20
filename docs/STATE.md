# State

_Last updated: 2026-08-20 (session 9 — Level 3 benchmark run, model selected)_

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
- **Stack confirmed and portability risks registered** (D35–D41).
  - **PyTorch removed from the dependency tree.** `silero-vad` imports torch
    unconditionally; a second CUDA runtime on a 4 GB card is a real threat
    (D18). Silero now runs from a vendored `silero_vad.onnx` via onnxruntime,
    measured at **0.156 ms per 32 ms frame** — 6× cheaper than D23 estimated.
  - Demo machine is the only target; **no CPU fallback** will be built (D36).
  - Translation over **REST + API key**, not the gRPC client library (D37).
  - **FastAPI + uvicorn**; UI is vanilla HTML/CSS/JS, no npm, no build step (D38).
  - **`uv` + committed `uv.lock`** replaces `requirements.txt` — one lockfile
    resolving both platforms is what makes "works on my machine" testable (D39).
  - `doctor` preflight command, built in step 1 (D40).
  - Portability register with mitigations (D41) — including that the Windows
    console is cp1252 and will crash on printing a non-ASCII caption.
- Repo initialised, docs written.
- **Level 0 (Foundation) built and passing on Linux.** First application code.
  - `pyproject.toml` + committed `uv.lock` — 45 packages, **zero torch**, and
    every package has a `win_amd64` wheel or an sdist. `required-environments`
    pins both platforms at lock time, so a Linux-only resolution cannot pass
    silently (D39).
  - `speech_translator/` package: `config.py` (D25 values, `model_size` and
    `compute_type` deliberately `None` until Level 3), `logging_setup.py`
    (UTF-8 forced at package import, so no entry point can forget it),
    `windows_cuda.py`, `doctor.py`, `__main__.py` stub.
  - `assets/silero_vad.onnx` vendored with its MIT license and a provenance
    note; `doctor` re-measures it — **0.109–0.150 ms per 32 ms frame** here,
    consistent with D35's 0.156. *(Session 7: that timing was taken at the wrong
    input width and the check could not fail on a dead model — corrected in D53.
    At the right width it is 0.116–0.141 ms, still inside the budget.)*
  - `doctor` runs 14 checks. On Linux: 7 pass, 3 expected-fail (CUDA, cuDNN
    DLLs, WASAPI loopback), 1 warn (no model chosen yet), 1 real fail (no API
    key yet). Expected failures are labelled and excluded from the exit code.
  - 16 tests pass on Linux with no GPU, no audio hardware, no network.
- **Level 1 (Audio capture) built and passing on Linux.** The first pipeline
  stage.
  - `speech_translator/audio/`: `AudioSource` Protocol exactly as
    `INTERFACES.md` §1 states it, `FrameFormatter` (decode → downmix →
    streaming resample → 512-sample frames, all inside the source per D24),
    `WavFileSource`, `WasapiLoopbackSource`, and `open_source()` — the one
    place platform detection is allowed to appear.
  - `speech_translator/tools/`: `list_devices`, `record_loopback`.
  - **The chunk-seam risk is closed and mutation-tested (D46).** One
    `soxr.ResampleStream` per source, never rebuilt. Same 10 s signal, ragged
    chunks: out-of-band energy **−81.4 dB** streaming vs **−30.2 dB** with a
    per-chunk resampler. Injecting the bug into `FrameFormatter` fails exactly
    two tests and no others.
  - **16 kHz mono int16 is bit-exact by construction** — no resampler is built,
    no downmix runs, no float round-trip. So replaying our own recordings is
    identical to the capture, not resampled twice.
  - `record_loopback` writes the *normalised* stream, because `BENCHMARK.md`
    consumes it at Level 3 and because that makes the recording itself the
    evidence for the format layer (D47). `--input-wav` runs the whole tool on
    Linux, leaving device open as the only genuinely Windows-only part (D48).
  - 34 new tests; **50 pass** on Linux with no GPU, no audio hardware, no
    network.
- **Level 2 (Segmentation) built and passing on Linux.** Utterances, not frames.
  - `speech_translator/segment/`: `SileroVad` over the vendored ONNX, the
    `SpeechDetector` Protocol it satisfies, `Utterance` exactly as
    `INTERFACES.md` §2 defines it, and `Segmenter` — close on silence or
    max-length (D12) at the D25 values, with the D28 runtime override built
    ahead of its caller. `tools/dump_utterances` is the inspection instrument.
  - **The VAD was silently dead and nothing said so (D53).** The vendored file
    is Silero **v5**, which needs 64 samples of the previous frame prepended to
    the 512 new ones — a 576-wide call. `INTERFACES.md`, D35, `assets/README.md`
    and `doctor` all said 512. The ONNX input dim is dynamic, so the wrong call
    runs and returns **0.0005 on real recorded speech** where the right one
    returns **1.0**. `doctor` passed it, because it only checked that the session
    ran. All four are corrected; `doctor` now asserts speech > 0.9 and silence
    < 0.1, so a dead VAD fails the preflight.
  - Two pieces of state carry across a frame boundary, not one — the LSTM state
    and the audio context. Both have a guard **and a control proving the guard
    can see the bug**, the same discipline D46 applied one layer down.
  - **Two labelled TUNABLE decisions**, not derived: thresholds 0.50 open /
    0.35 release, with hysteresis rather than a single cut (D54), and pre-roll
    128 ms / tail pad 192 ms (D55). No minimum-speech-run, deliberately — D30's
    discard already covers blips and a min-run would spend latency budget.
  - The 160 ms `speaking` debounce is a measurement: at 96 ms the indicator
    blinked dark mid-phrase on the fixture.
  - `assets/speech_fixture.wav` (13.9 s, generated with espeak-ng, committed with
    provenance in `assets/README.md`) exists because **nothing numpy can
    synthesise reliably crosses Silero's threshold** (D56) — so without speech in
    the repo the whole stage can be dead and green.
  - 54 new tests; **104 pass** on Linux with no GPU, no audio hardware, no
    network.
- **Level 3 benchmark run on Windows (2026-08-20).** Model selected.
  - All seven configurations ran for 10 sustained wall-clock minutes each on the
    GTX 1650 (4096 MB, driver 592.82). Total run ~75 minutes.
  - **Two configs pass gate 1 (RTF < 0.5):** small:int8_float16 (0.279) and
    medium:int8_float16 (0.488). All float16 variants fail — dramatically so
    (1.58–2.62× real-time).
  - **Selected: medium:int8_float16.** RTF 0.488, peak VRAM 1849 MB, no thermal
    throttling (-10.6% — gets faster over time). Transcription quality is
    sufficient for translation input (small garbles proper nouns and
    code-switches, making it unsuitable).
  - **int8_float16 massively beats float16 on this no-tensor-core GPU:** 2.3× for
    small, 3.2× for medium, 4.3× for large-v3-turbo. This is the D18 open
    question answered: the int8 path is bandwidth-bound, not compute-bound.
  - Gate 3 (word age): unknown — no API key was present, so MT latency was not
    measured. At assumed 300 ms: predicted 6852 ms < 7000 ms (would pass).
  - **DLL trap fixed (D60).** `os.add_dll_directory()` alone does not make cuBLAS
    findable by CTranslate2 at inference time — PATH prepend also needed.
  - No headroom for diarization (D9 remains descoped).
  - `config.py` updated: `model_size = "medium"`, `compute_type = "int8_float16"`.
  - Harness code unchanged from session 8. Full results in
    `var/benchmark/20260820-145519/`.

## Next

**Level 4 — Transcription.** The gate is passed and the model is chosen. Build
the transcription worker process: spawn it, load medium:int8_float16 on CUDA,
feed it utterances from the queue, return timestamped text. See `PLAN.md` for
acceptance criteria.

| Level | | Status |
|---|---|---|
| 0 | Foundation — skeleton, `uv.lock`, vendored ONNX, `doctor` | **done** |
| 1 | Audio capture — `AudioSource`, WASAPI + WAV, capture tools | **done** |
| 2 | Segmentation — Silero VAD, `Utterance` | **done** |
| 3 | **Benchmark (gate)** — picks model + `compute_type` | **done** — medium:int8_float16, RTF 0.488 |
| 4 | Transcription — worker process, hallucination guard, LID | |
| 5 | Sentences + translation — carry-over, flush, budget | |
| 6 | Server + UI — FastAPI, six states, caption cards | |
| 7 | Tuning + evidence — p95 word age measured | |
| 8 | Demo + journey doc | |

**Do not duplicate `PLAN.md` here.** This file records status; that one records
the plan. When they disagree, `PLAN.md` is wrong or the work is.

**Note on ordering:** the benchmark used to be step 1 and marked as gating
everything. It moved because CUDA answered the existential question it was asked
to answer — see D20. Levels 0-2 are model-independent, so building them first
costs nothing and gives the benchmark a real capture path to source its sample
audio from.

## In progress
Nothing. Level 3 is complete. Level 4 (Transcription) is next.

## Blocked
- **Three truncated lines in the assignment brief** are still unknown — see
  `PRD.md`. The scope line gates whether incoming-only is permitted. Raised
  again in session 4 and still unanswered; it is the only open item that can
  invalidate work already designed.
- **Deadline unknown.** The journey doc is due 48h before the interview, so the
  real deadline is earlier than the interview date.
- **MT round trip unmeasured.** Gate 3 reported "unknown" because no Google
  Translate API key was present during the benchmark. At assumed 300 ms the
  prediction passes (6852 ms < 7000 ms), but the measured value is owed at
  Level 7. See `docs/API_MIGRATION_NOTE.md` for alternatives considered.

## Known environment quirks (dev box only)
- The Linux laptop exports `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`
  globally. pytest autoloads a ROS plugin from it and dies before collection.
  Run tests as `PYTHONPATH= uv run pytest`. Deliberately **not** worked around in
  project config — it is a property of this machine, not of the project, and
  hard-coding a ROS-specific opt-out into `pyproject.toml` would outlive the
  reason for it.

## Open questions
- Which reading of the silence/punctuation bonus is intended (correct
  punctuation in output vs. visibly showing pauses in the UI)? See `PRD.md`.
  The design covers both cheaply — `preceding_silence_ms` reaches the UI on the
  first sentence of every utterance — so this is no longer urgent.
- Desktop shell (Electron/Tauri) around the browser UI — deferred, revisit once
  the core pipeline works.
- ~~Whether `int8_float16` beats `float16` on a GPU with no tensor cores~~ —
  **answered: yes, dramatically.** 2.3× for small, 3.2× for medium, 4.3× for
  large-v3-turbo. The int8 path is bandwidth-bound; halving weight size halves
  the bandwidth pressure regardless of tensor cores.
- ~~Whether Level 3 runs before Level 2~~ — moot, Level 2 is built (D52 stands
  as the record of the argument).
- Whether the D54/D55 values survive real meeting audio. They are TUNABLE and
  were chosen against generated speech; the falling-edge and dip numbers that
  motivated hysteresis came from real recorded speech, but the 0.50/0.35 pair,
  the 160 ms debounce and the 128/192 ms padding all want a pass at Level 7.
- ~~Whether Level 3's benchmark should chunk at the *real* utterance-length
  distribution~~ — settled in D57: it does, cut once and shared across all seven
  rows, with D52's fixed 4 s / 1.5 s bracket kept as a check on the selected
  configuration. Still open underneath it: the chunk distribution depends on the
  TUNABLE VAD thresholds (D54, D55), so a retune at Level 7 moves the RTF
  denominator. The distribution is recorded in `results.json` so that comparison
  is possible rather than guesswork.
- ~~Whether the demo machine's loopback endpoint negotiates `paInt16` or falls
  back to `paFloat32`, and what `defaultSampleRate` it reports~~ — verified in
  session 9 (Gemini): recording captured successfully at 16 kHz mono int16.
  Exact device negotiation details in the chat export from that session.
- Sentence-boundary splitting for languages without Latin punctuation
  conventions (e.g. Thai, which does not space or full-stop the same way). Not a
  problem for the likely demo languages; note it before claiming generality.
- **Demo plan** — which audio is played, in which language, and what happens if
  the network drops mid-demo (the MT cache from D31 covers repeats only). Not
  yet discussed.
- **Journey-doc screenshot plan** — it is a graded deliverable and screenshots
  are easiest to capture as each stage lands, not retroactively.
