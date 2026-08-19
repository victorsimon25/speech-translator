# State

_Last updated: 2026-08-19 (session 8 — Level 3's harness built)_

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
- **Level 3's harness built and dry-run on Linux.** The code half of the gate.
  **No benchmark numbers exist**, and none can until the recording does.
  - `speech_translator/tools/benchmark.py` executes `BENCHMARK.md`: seven
    configurations, per-chunk timing, warm-up and model load excluded from RTF,
    first-minute vs last-minute split, `nvidia-smi` polled at 1 Hz, the three
    gates evaluated, and `results.json` + `results.md` written **incrementally**
    so a configuration that OOMs is a failed row and the matrix carries on (D51).
  - **Three decisions, argued rather than defaulted** (D57–D59): chunks come from
    the real `Segmenter` now that Level 2 exists, cut **once** and shared
    byte-identically by all seven rows, with D52's fixed 4 s / 1.5 s bracket kept
    for the selected row; the dry run uses a stub engine by default and a real
    CPU model behind a flag; and "ten sustained minutes" is ten minutes of **wall
    clock** with the audio looped, which is the only reading under which
    7 × 10 min is the ~70 minutes D51 budgeted.
  - **`MT_roundtrip` is measured, never assumed.** Ten live calls, p95. With no
    API key the term is recorded as *missing* and gate 3 reports `unknown`; the
    guessed 300 ms is not substituted anywhere, and there is a test for that.
    Gate 3's column says **predicted** in the JSON, the markdown and the console.
  - The CUDA guard refuses to time anything until `get_cuda_device_count()`
    returns 1, and names D36 when it does not — a silent CPU fallback would
    produce seven rows of fiction rather than a slow run.
  - 49 new tests; **153 pass** on Linux with no GPU, no weights, no network.
    Two real invocations besides: the stub engine, and `faster-whisper tiny`
    decoding on CPU (RTF 0.137 over 82 chunks, correct transcripts) — which is
    what proves the lazy `segments` generator is drained inside the timed region
    rather than timed empty.
  - **What the harness cannot prove here:** that a GPU answers. VRAM polling,
    thermal throttling, a real OOM and `get_cuda_device_count() == 1` are all
    owed to the demo machine, and so is the input — see Blocked.

## Next

**Level 3 — Benchmark (gate), on Windows.** The harness is built; what is left
is the measurement, and it is blocked on Level 1's 10-minute recording, which is
its input. See [`PLAN.md`](PLAN.md) for the full ladder and each level's
acceptance criteria.

**[`WINDOWS.md`](WINDOWS.md) is the runbook for that session** — the commands
below in full, what to record, what to do when the DLL trap bites, and the list
of docs to update afterwards. It also carries the prompt to start the session
with.

**On the Windows machine, now four levels' worth and no longer just a
formality:**

```
git pull
uv sync                                                  # Level 0, still owed
python -m speech_translator.doctor                       # Level 0, still owed
python -m speech_translator.tools.list_devices           # Level 1
python -m speech_translator.tools.record_loopback -t 600 # Level 1 + BENCHMARK input
python -m speech_translator.tools.dump_utterances \
    --input-wav <that recording> --write-wav utts/       # Level 2, the open box
python -m speech_translator.tools.benchmark \
    --input-wav <that recording>                         # Level 3, ~70 minutes
```

Pre-download the Whisper weights before the timed run, or the first
configuration's `load_s` is mostly a download. Open a browser first — gate 2 is
about the card the browser is already drawing from (D18). Expect the harness to
refuse to start if `get_cuda_device_count()` is not 1; that refusal is the
feature.

Then **listen to `utts/`**. That is Level 2's one open acceptance criterion —
boundaries landing at real pauses on real meeting audio — and it costs nothing
extra once the recording exists.

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
| 2 | Segmentation — Silero VAD, `Utterance` | **done (Linux); real-speech inspection owed** |
| 3 | **Benchmark (gate)** — picks model + `compute_type` | **harness built (Linux); the measurement is blocked on Level 1's recording** |
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
Nothing. Levels 0, 1 and 2 are complete on Linux and Level 3's harness is built;
their Windows verification is the first task of the next session on that machine,
and all four fit in one sitting after a single `git pull` — though Level 3's own
runs are 70 minutes of it.

## Blocked
- **Level 0's Windows half is unverified.** `uv sync` and `doctor` have not run
  on the demo machine. Not blocking Level 2, which is Linux work, but it is
  blocking in the sense that the lockfile's cross-platform claim is currently
  argued rather than demonstrated. Unchanged since session 5 — do not let it
  quietly drop off this list.
- **Level 2's real-speech inspection is unverified**, and it is the same
  blocker as the one below: `PLAN.md`'s "on a captured meeting WAV, boundaries
  land at real pauses" needs a captured meeting WAV. The generated fixture (D56)
  covers the wiring and the state machine, and `dump_utterances --write-wav`
  makes the check a five-minute job once the recording exists — but TTS speech
  has no room tone, no music bed and no overlap, which is exactly what Silero is
  in the design to survive. Do not let the fixture be mistaken for the criterion.
- **Level 1's Windows half is unverified.** No WASAPI endpoint has been opened.
  The format layer, both sources and both tools are tested here, and the
  chunk-seam failure is measured and mutation-tested (D46), but "captures 10
  minutes that plays back cleanly" is a claim about hardware and stays open.
- **One missing recording now blocks four things at once.** The 10-minute
  `record_loopback` capture is the single highest-value thing the Windows machine
  can produce, and it is worth seeing the list in one place rather than spread
  across four bullets:

  | Blocked | Needs |
  |---|---|
  | Level 0's Windows half | `uv sync` + `doctor` — same sitting, no recording needed |
  | Level 1's playback check | the recording, listened to |
  | Level 2's boundary inspection | `dump_utterances --write-wav` over that recording |
  | **Level 3's entire measurement** | that recording as `--input-wav` |

  One session on the demo machine clears all four. Nothing about it is hard; it
  has simply not happened, and it has been outstanding since session 5.
- **Level 3 has produced no numbers, and must not appear to have.** The harness is
  built and tested; `BENCHMARK.md`'s hardware survey and results table are
  deliberately still empty, and none of Level 3's acceptance boxes are ticked. A
  13.9 s generated fixture looped 20 times exercises the machinery and is not a
  benchmark result.
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
