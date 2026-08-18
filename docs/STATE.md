# State

_Last updated: 2026-08-18 (session 6 — Level 1 built)_

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

## Next

**Level 2 — Segmentation.** See [`PLAN.md`](PLAN.md) for the full ladder and
each level's acceptance criteria.

**On the Windows machine, now two levels' worth and no longer just a
formality:**

```
git pull
uv sync                                                  # Level 0, still owed
python -m speech_translator.doctor                       # Level 0, still owed
python -m speech_translator.tools.list_devices           # Level 1
python -m speech_translator.tools.record_loopback -t 600 # Level 1 + BENCHMARK input
```

Expect CUDA, cuDNN and loopback to flip to PASS. Three things to write down
because they are unknown from here: whether `uv sync` agrees with the lockfile,
what `defaultSampleRate` the endpoint reports, and whether it negotiates
`paInt16` or falls back to `paFloat32` (D49). Then **listen to the recording** —
correct pitch and no clicks is Level 1's last open acceptance box, and that
recording is also Level 3's benchmark input, so it is wanted anyway.

| Level | | Status |
|---|---|---|
| 0 | Foundation — skeleton, `uv.lock`, vendored ONNX, `doctor` | **done (Linux); Windows run owed** |
| 1 | Audio capture — `AudioSource`, WASAPI + WAV, capture tools | **done (Linux); Windows run owed** |
| 2 | Segmentation — Silero VAD, `Utterance` | **next** |
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
Nothing. Levels 0 and 1 are complete on Linux; their Windows verification is the
first task of the next session on that machine, and it is now a single `git pull`
away from being done in one sitting.

## Blocked
- **Level 0's Windows half is unverified.** `uv sync` and `doctor` have not run
  on the demo machine. Not blocking Level 2, which is Linux work, but it is
  blocking in the sense that the lockfile's cross-platform claim is currently
  argued rather than demonstrated. Unchanged since session 5 — do not let it
  quietly drop off this list.
- **Level 1's Windows half is unverified.** No WASAPI endpoint has been opened.
  The format layer, both sources and both tools are tested here, and the
  chunk-seam failure is measured and mutation-tested (D46), but "captures 10
  minutes that plays back cleanly" is a claim about hardware and stays open.
  It is also the input `BENCHMARK.md` needs, so **Level 3 cannot start until
  this recording exists.**
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
- Whether Level 3 runs before Level 2. It may (D52) — the benchmark chunks at a
  fixed 4 s and needs no Segmenter — at the cost of an optimistic RTF, mitigated
  by bracketing at 1.5 s. Not yet chosen; the deciding factor is how scarce
  Windows access is.
- Whether the demo machine's loopback endpoint negotiates `paInt16` or falls
  back to `paFloat32`, and what `defaultSampleRate` it reports. Both paths are
  built and both feed the same formatter (D49), so this is a fact to record on
  the first Windows run, not a risk.
- Sentence-boundary splitting for languages without Latin punctuation
  conventions (e.g. Thai, which does not space or full-stop the same way). Not a
  problem for the likely demo languages; note it before claiming generality.
- **Demo plan** — which audio is played, in which language, and what happens if
  the network drops mid-demo (the MT cache from D31 covers repeats only). Not
  yet discussed.
- **Journey-doc screenshot plan** — it is a graded deliverable and screenshots
  are easiest to capture as each stage lands, not retroactively.
