# Build Plan

The ladder from an empty repo to a demonstrable product. **One level at a time**;
each level's acceptance criteria must pass before the next begins.

- `STATE.md` says *where we are*. This file says *where we are going*.
- `INTERFACES.md` says *what each stage's contract is*. This file says *when it
  gets built and how you know it works*.
- Every level names what it **unblocks**, so the cost of skipping one is visible.

**Where each level can be built** matters because of D22 — written on Linux, run
on Windows, git as the bridge. A level marked *Windows* cannot be verified on the
dev laptop no matter how finished the code looks.

| Level | Name | Build on | Status |
|---|---|---|---|
| 0 | Foundation | both | **done on Linux; Windows half unverified** |
| 1 | Audio capture | Linux code / Windows verify | **done on Linux; Windows half unverified** |
| 2 | Segmentation | Linux | **done on Linux; real-speech inspection owed to Windows** |
| 3 | **Benchmark (gate)** | harness on Linux / **measurement Windows only** | **harness built and dry-run on Linux; the measurement is owed to Windows** |
| 4 | Transcription | Linux code / Windows run | not started |
| 5 | Sentences + translation | Linux | not started |
| 6 | Server + UI | Linux (fake ASR) / Windows real | not started |
| 7 | Tuning + evidence | **Windows only** | not started |
| 8 | Demo + journey doc | Windows | not started |

---

## Level 0 — Foundation

**Goal:** a repo that installs identically on both machines and can tell you what
is missing on either.

**Deliverables**
- `pyproject.toml` + committed `uv.lock` (D39). Platform deps carry
  `sys_platform == "win32"` markers (D22).
- `speech_translator/` package; `config.py` holding the D25 values.
- Vendored `assets/silero_vad.onnx` (2.3 MB, MIT). **No torch, ever** (D35).
- `.env.example`, `.gitattributes` (`* text=auto eol=lf`), UTF-8 forced on all
  stream handlers and log files (D41).
- `python -m speech_translator.doctor` (D40).

**Acceptance**
- [x] `uv sync` succeeds on Linux **and** Windows from the same lockfile.
      *Linux: verified, 40 packages. Windows: not yet run — the lock carries a
      `win_amd64` wheel or an sdist for all 45 packages (checked
      programmatically), and `required-environments` makes that a lock-time
      guarantee, but the run itself is owed on the demo machine.*
- [x] `import speech_translator` succeeds on Linux (D22 — no eager Windows imports).
- [x] `doctor` runs on both and reports honestly: on Linux, CUDA and loopback
      report **fail** and that is the correct output, not a bug.
      *Linux verified; Windows run owed. Expected failures are labelled and do
      not set the exit code, so a red line on the dev box stays distinguishable
      from a red line on the demo machine.*
- [x] `pip list | grep torch` returns nothing. *Also: the string `torch` does not
      appear anywhere in `uv.lock`.*

**Carried to Level 1** (needs the Windows machine, nothing here blocks on it):
`uv sync` + `doctor` on Windows. Until then Level 0 is done on one of its two
platforms.

**Unblocks** everything. **Risk:** low.

---

## Level 1 — Audio capture

**Goal:** 16 kHz mono int16 frames coming out of a real meeting, and a WAV file
of the same for offline work.

**Deliverables**
- `AudioSource` Protocol; frame constants (16 kHz, mono, int16, 512 samples).
- Format layer: decode → downmix → **streaming** resample (`soxr.ResampleStream`,
  so filter state carries across chunks) → exact 512-sample frames. Inside the
  source (D24).
- `WavFileSource` (realtime and as-fast-as-possible) — tests + Linux smoke path.
- `WasapiLoopbackSource` — lazy `pyaudiowpatch` import (D22).
- `tools/list_devices`, `tools/record_loopback`.

**Acceptance**
- [x] Every emitted frame is exactly 1024 bytes; total sample count matches
      source duration to within one frame. *Verified at 16 k/44.1 k/48 k/96 k,
      mono and stereo, with ragged 1–5000-sample chunks. The tolerance is one
      frame because `flush()` zero-pads the tail — Silero takes 512 samples or
      nothing (D23), so a partial tail is completed, not dropped.*
- [x] A 48 kHz stereo WAV and a 16 kHz mono WAV both produce identical-shaped
      output; the 16 kHz mono path is bit-exact (no needless float round-trip).
      *Bit-exact by construction, not by tolerance: at 16 kHz mono int16 no
      resampler is constructed, no downmix runs and no float conversion happens
      (D46). The test asserts byte equality with the source.*
- [ ] `record_loopback` on Windows captures 10 minutes of meeting audio that
      plays back cleanly — correct pitch, no clicks at chunk boundaries.
      **Owed to the Windows machine.** *The half that can be checked here has
      been: a 10 s 48 kHz stereo signal through the real tool lands its spectral
      peaks at exactly 1000 and 2500 Hz (pitch correct, no rate-ratio error) with
      out-of-band energy 81.4 dB down. The same signal through a per-chunk
      resampler — the failure this criterion is really about — sits at 30.2 dB.
      What is genuinely untested is the device: opening the endpoint, the
      negotiated sample format, and ten sustained minutes of it.*
- [x] Unit tests pass on Linux with no audio hardware. *34 new tests, 50 total,
      no GPU, no network, no device.*

**Also carried to Windows:** `list_devices` against real hardware, and which of
`paInt16` / `paFloat32` the demo machine's endpoint actually negotiates (D49).

**Unblocks** Level 2, and **the benchmark's sample audio** — `BENCHMARK.md`
sources its input through `record_loopback`, so this is on the critical path.

**Risk:** medium. Device format variance; clicks if the resampler is
re-instantiated per chunk. *The second risk is closed on the code side and
mutation-tested (D46); the first is what the Windows run is for.*

---

## Level 2 — Segmentation

**Goal:** utterances, not frames.

**Deliverables**
- Silero wrapper over the vendored ONNX via `onnxruntime`, state carried between
  frames (D35). **Two kinds of state, as it turned out** — the LSTM state and a
  64-sample audio context (D53).
- `Segmenter` — close on silence **or** max-length (D12) at the D25 values;
  accepts a **runtime override** of effective `max_utterance_ms` so Level 6's
  controller can shrink it (D28).
- `min_utterance_ms` discard (D30); `preceding_silence_ms`; `speaking` side channel.
- `tools/dump_utterances`.
- *Added during the level:* `assets/speech_fixture.wav` and its generator, because
  no positive VAD test is possible without speech in the repo (D56); the two
  TUNABLE threshold decisions (D54) and the pre-roll/tail-pad decision (D55).

**Acceptance**
- [ ] On a captured meeting WAV, boundaries land at real pauses on inspection.
      **Owed to the Windows machine** — that WAV does not exist yet, because
      `record_loopback` has not run on the demo machine (Level 1's open box).
      *What has been checked here: on the generated speech fixture, whose
      silences are 900–1000 ms by construction, boundaries land inside those
      pauses and nowhere else, and each utterance is written out as its own WAV
      by `dump_utterances --write-wav` and can be listened to. That is a stronger
      check than "it ran" and a weaker one than the criterion: TTS speech has no
      room tone, no music bed and no overlap, which is what Silero is here to
      survive (D23, D56).*
- [x] No utterance exceeds `max_utterance_ms`. *Asserted for every utterance in
      every test, including 22 s of unbroken speech. The cap floors to a whole
      frame, so it holds at the 2000 ms backpressure floor too — 1984 ms, not
      2016 (D55).*
- [x] Silence-only and music-only input produce **zero** utterances. *Silence, a
      440 Hz tone, a four-note chord and white noise, each 6 s through the real
      Silero: nothing above 0.05, zero utterances.*
- [x] Sub-300 ms blips are discarded, not emitted. *And measured on speech, not
      on the padded duration — the pre-roll and tail pad are 320 ms together, so
      testing the padded duration would have cancelled the guard entirely (D55).*
- [x] VAD cost stays near the measured 0.156 ms/frame. *0.116–0.141 ms per 32 ms
      frame at the corrected 576-sample width, single threaded, CPU only.
      `dump_utterances` reports it per run.*
- [x] *(added)* The VAD actually detects speech, and a guard fails if it stops.
      *This was not on the list, and it is the one that mattered: the documented
      512-sample call returns 0.0005 on real speech and no error (D53).*

**Unblocks** Level 3 (realistic chunk lengths) and Level 4.

**Risk:** medium — this is one of the two places tuning time goes (D12).

---

## Level 3 — Benchmark  ⛔ GATE

**Goal:** the two config values that Level 4 cannot be written without.

Not code. A measurement, run per `BENCHMARK.md` on the Windows machine, with a
browser open (D18).

**Deliverables**
- **`tools/benchmark.py`** — the harness. **Built.** 7 configurations × 10
  minutes is ~70 minutes of runs with per-chunk timing, VRAM polling and
  first/last-minute RTF split; that is not a job for a REPL. **Written and
  dry-run on Linux against `WavFileSource`, executed on Windows** (D51) — the
  only Windows-specific thing in it is that a GPU answers. Emits the results row
  as JSON so the table is transcribed, not retyped.
  - *Added during the level:* chunking at the Segmenter's real utterance
    distribution rather than a fixed 4 s, now that Level 2 exists (D57); a stub
    engine so the dry run lives in `pytest` (D58); and "ten sustained minutes"
    resolved as ten minutes of **wall clock** with the audio looped (D59).
- Hardware survey table filled in (driver, CUDA, VRAM at idle, CPU/RAM).
- 7 configurations × 10 minutes sustained, first-minute vs last-minute RTF.
- Filled results table including the **p95 word age (predicted)** column, and a
  **measured** MT round trip feeding it rather than a guessed 300 ms.
- Selected `model_size` + `compute_type` written into config.

**Acceptance**

Every box below is a **measurement**, and all five are owed to the Windows
machine. The harness that will take them is built and tested; that is not the
same thing, and this level does not advance until the recording exists and the
runs happen.

- [ ] `ctranslate2.get_cuda_device_count()` returns `1` before any timing is trusted.
      *The guard is built and tested — `require_cuda()` refuses to time anything
      and names D36 in the message — but a count of 1 has never been observed
      from this repo. On Linux it returns 0, which is the correct answer here.*
- [ ] At least one configuration passes **all three** gates: sustained RTF < 0.5,
      VRAM leaves room for desktop + browser, **p95 word age < 7 s** (predicted
      here from measured RTF and a measured MT round trip; Level 7 measures it
      end to end — see `BENCHMARK.md` gate 3 and D51).
      *Gate arithmetic is implemented and tested against hand-computed cases,
      including that an unmeasured MT term leaves gate 3 `unknown` rather than
      substituting 300 ms. No configuration has been run on a GPU.*
- [ ] The chosen config is the *fastest that is accurate enough*, not the largest
      that fits (D26). *Not automatable and deliberately not automated: "enough"
      is a listening test. The harness keeps sample transcripts in
      `results.json` for that judgement and prints that it is not making it.*
- [ ] Throttling penalty (first vs last minute) recorded.
      *The split is implemented and tested — including that the warm-up chunk is
      excluded from the first minute, which it would otherwise dominate. A real
      penalty needs a GPU that throttles.*
- [ ] The harness runs on Linux against a WAV before it is trusted on Windows —
      a benchmark whose first execution is on the machine that matters is a
      benchmark you are debugging instead of running.
      **Partially met, and the remainder is honest to state.** *It runs here
      against `assets/speech_fixture.wav`: 49 tests, plus two real invocations —
      one with the stub engine, one with `faster-whisper tiny` decoding on CPU
      (RTF 0.137 over 82 chunks, load excluded, warm-up separate, transcripts
      correct). What it has **not** run against is the input it was written for:
      the 10-minute `record_loopback` capture does not exist yet. A 13.9 s
      fixture looped 20 times exercises the machinery; it does not exercise a
      real utterance-length distribution.*

**Unblocks** Level 4. **Nothing after this level can start without it.**

**Risk:** high — the cuDNN/cuBLAS DLL trap (D41), and the possibility that
nothing clears all three gates, in which case `BENCHMARK.md`'s fallback ladder
applies.

---

## Level 4 — Transcription

**Goal:** utterances become punctuated, language-tagged text.

**Deliverables**
- Worker **process** with `spawn` ready-handshake before the UI reports running
  (D27). Bounded IPC queue, `maxsize = 4`.
- `faster-whisper` at the benchmark's config; `beam_size=1`,
  `condition_on_previous_text=False`, `vad_filter=False`.
- `Transcript` incl. `no_speech_prob` / `avg_logprob`; hallucination guard (D30).
- Confidence-weighted LID over the first ~10 s of speech, then lock (D32).
- JSONL timing log (D34).

**Acceptance**
- [ ] WAV in → correct punctuated text out, correct language detected.
- [ ] Near-silence and noise produce **no** caption rather than "Thank you."
- [ ] Worker load failure surfaces as `error` state, not a hang.
- [ ] Per-utterance RTF appears in the JSONL log.

**Unblocks** Level 5. **Risk:** medium-high — first CUDA-in-anger level.

---

## Level 5 — Sentences and translation

**Goal:** caption-sized, translated text, in order, within budget.

**Deliverables**
- `SentenceSplitter` — carry-over fragment **and the flush rule** (D29).
- `Translator` over REST + API key (D37), behind an interface.
- Single serialized translation task preserving order (D27).
- LRU cache; persisted monthly character counter; `src == tgt` skip (D31).
- Failure path → `target_text = null` + `mt_error`, never a stall.

**Acceptance**
- [ ] A mid-sentence max-length cut recombines into one correct sentence.
- [ ] A held fragment **is emitted** on stop / long silence — the last words of a
      session must appear.
- [ ] Captions arrive in spoken order under concurrent translation.
- [ ] Character counter survives a process restart.
- [ ] Forced API failure degrades to source-only without blocking the pipeline.

**Unblocks** Level 6. **Risk:** medium. The flush rule is the easy thing to miss.

---

## Level 6 — Server and UI

**Goal:** the thing you actually demonstrate.

**Deliverables**
- FastAPI + uvicorn serving static UI **and** WebSocket in one process (D38).
- Full message protocol per `INTERFACES.md` §6.
- Backpressure controller: adaptive shrink → drop oldest, with `dropped` events (D28).
- UI: six defined states (PRD), caption cards translation-primary /
  source-secondary (D33), speaking indicator, health badge, language chips with
  override, gap markers from `preceding_silence_ms`, auto-scroll with jump-to-live,
  transcript export.

**Acceptance**
- [ ] End to end: loopback audio → captions in the browser.
- [ ] All six states reachable and visually distinct.
- [ ] Captions never rewrite themselves (D11).
- [ ] `word_age_ms` and `rtf` both visible.
- [ ] Killing the network mid-session degrades to source-only captions, visibly.
- [ ] UI is usable with no explanation — test by handing it to someone cold.

**Buildable on Linux** against a fake transcriber, which is worth doing: UI work
is the most iteration-heavy and the dev loop should not be push-pull-run.

**Risk:** medium — scope creep lives here.

---

## Level 7 — Tuning and evidence

**Goal:** turn "it works" into "here are the numbers".

**Deliverables**
- 10-minute sustained live run on the demo machine, browser open.
- JSONL log → latency histogram, RTF curve, queue-depth graph, **p95 word age**.
- Tuning pass on `silence_threshold_ms` / `max_utterance_ms` against real speech.
- `BENCHMARK.md` and `PRD.md` DoD updated with measured results.

**Acceptance**
- [ ] p95 word age < 7 s **measured end to end**, not inferred from RTF.
- [ ] Queue depth stays at 0–1 for the whole run.
- [ ] No CUDA OOM across 10 minutes with a browser open.
- [ ] Every PRD Definition-of-Done box ticked, each with evidence.

**Risk:** medium — this is where a passing benchmark meets real speech.

---

## Level 8 — Demo and journey doc

**Goal:** the graded deliverable.

**Deliverables**
- Demo script: which audio, which language pair, what to say about the tradeoffs.
- Rehearsal on the demo machine, including the failure fallbacks.
- Journey doc assembled from `docs/journey/` with screenshots and prompt logs.

**Acceptance**
- [ ] Demo runs start to finish twice without intervention.
- [ ] Journey doc delivered **48 h before the interview**.
- [ ] A stated answer for: what happens if the network drops mid-demo.

**Risk:** the deadline is still unknown (see `STATE.md` → Blocked).

---

## Traceability — which level satisfies which DoD item

| PRD Definition of Done | Level |
|---|---|
| Captures system audio on Windows | 1 |
| Source language auto-detected, displayed, overridable | 4 (detect) + 6 (UI) |
| Target language selectable | 6 |
| Captions do not rewrite themselves | 6 |
| p95 word age < 7 s | 7 (measured), 3 (predicted) |
| Output correctly punctuated | 4 |
| UI needs no explanation | 6 |
| Sustained RTF < 0.5 | 3 (predicted), 7 (confirmed) |
| Journey doc | 8, written continuously from level 0 |

## What is deliberately not on this ladder

Speaker diarization (D9/D21 — reopen only if Level 3 shows headroom), Linux
capture (D17), TTS (D5), microphone/two-way (D4), meeting-platform integrations
(D10), desktop shell (D13), scaling (D14).
