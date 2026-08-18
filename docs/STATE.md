# State

_Last updated: 2026-08-18 (session 5 — Level 0 built)_

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
    consistent with D35's 0.156.
  - `doctor` runs 14 checks. On Linux: 7 pass, 3 expected-fail (CUDA, cuDNN
    DLLs, WASAPI loopback), 1 warn (no model chosen yet), 1 real fail (no API
    key yet). Expected failures are labelled and excluded from the exit code.
  - 16 tests pass on Linux with no GPU, no audio hardware, no network.

## Next

**Level 1 — Audio capture.** See [`PLAN.md`](PLAN.md) for the full ladder and
each level's acceptance criteria.

**First, on the Windows machine:** `git pull && uv sync && python -m
speech_translator.doctor`. That is the other half of Level 0's acceptance and it
cannot be done from here (D22). Expect CUDA, cuDNN and loopback to flip to PASS;
if `uv sync` disagrees with the lockfile, that is a finding, not a formality.

| Level | | Status |
|---|---|---|
| 0 | Foundation — skeleton, `uv.lock`, vendored ONNX, `doctor` | **done (Linux); Windows run owed** |
| 1 | Audio capture — `AudioSource`, WASAPI + WAV, capture tools | **next** |
| 2 | Segmentation — Silero VAD, `Utterance` | |
| 3 | **Benchmark (gate)** — Windows only; picks model + `compute_type` | |
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
Nothing. Level 0 is complete on Linux; its Windows verification is the first
task of the next session on that machine.

## Blocked
- **Level 0's Windows half is unverified.** `uv sync` and `doctor` have not run
  on the demo machine. Not blocking Levels 1–2, which are Linux work, but it is
  blocking in the sense that the lockfile's cross-platform claim is currently
  argued rather than demonstrated.
- **Three truncated lines in the assignment brief** are still unknown — see
  `PRD.md`. The scope line gates whether incoming-only is permitted. Raised
  again in session 4 and still unanswered; it is the only open item that can
  invalidate work already designed.
- **Deadline unknown.** The journey doc is due 48h before the interview, so the
  real deadline is earlier than the interview date.

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
- Whether `int8_float16` beats `float16` on a GPU with no tensor cores. The
  benchmark will say; do not assume the RTX answer transfers (D18).
- Sentence-boundary splitting for languages without Latin punctuation
  conventions (e.g. Thai, which does not space or full-stop the same way). Not a
  problem for the likely demo languages; note it before claiming generality.
- **Demo plan** — which audio is played, in which language, and what happens if
  the network drops mid-demo (the MT cache from D31 covers repeats only). Not
  yet discussed.
- **Journey-doc screenshot plan** — it is a graded deliverable and screenshots
  are easiest to capture as each stage lands, not retroactively.
