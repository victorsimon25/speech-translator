# Decisions

Append-only. Newest at the bottom. Every entry states the *why*, because the
why is what a future session cannot re-derive.

---

## 2026-08-18 — Session 1 (design)

### D1. Capture system audio, not microphone or meeting-platform APIs
The user needs to understand what *other* people say. Their voices arrive as
audio played out of the speakers, so the operating system's audio output is the
correct tap point.

**Why this over alternatives:** system audio is the only approach that is
agnostic to the meeting platform. Zoom, Teams, Meet, Slack, Discord, and a
YouTube video all play through the OS, so one code path covers all of them.
Every other route is per-platform.

**Known limitation, accepted:** the captured stream is a single mixed waveform.
Speaker identity was discarded by the meeting client before the audio reached
the sound card and cannot be recovered downstream. See D9.

### D2. Linux capture via PipeWire monitor source — verified, not assumed
Ubuntu 24.04 with PipeWire 1.0.5. Confirmed by inspection that sink node 47
exposes `monitor_FL` / `monitor_FR` with `port.monitor = "true"`. Loopback
capture is available with no kernel module, virtual cable, or driver install.

`pw-record` and `wpctl` are installed. `pactl` and `pavucontrol` are **not** —
installing `pavucontrol` is recommended before demo day to make routing
inspectable.

### D3. Windows capture via PyAudioWPatch (WASAPI loopback)
The user has a Windows machine available for testing, so cross-platform support
is real rather than claimed. `PyAudioWPatch` is a maintained PyAudio fork with
WASAPI loopback support and prebuilt wheels covering Python 3.7–3.13.

**Why it matters:** both platforms are pure Python and both satisfy the same
`AudioSource` interface, so cross-platform costs one extra implementation rather
than a native build system.

### D4. Incoming direction only, behind a pluggable input interface
Translating what others say is the stated need. Outgoing (microphone) capture is
not built.

**Why the abstraction anyway:** it costs almost nothing now, it gives a
`WavFileSource` for deterministic tests without live audio, and if the truncated
brief line turns out to require two-way, it becomes a config change rather than
a rewrite.

### D5. Text output only, no TTS
Stated requirement. Also removes a synthesis stage from the latency budget.

### D6. Local Whisper on CPU for transcription
Follows from the free-tier constraint. Speech APIs bill per audio-minute and
their free tiers are thin, so transcription must run locally. The machine has no
NVIDIA GPU (Intel UHD 620 only), so "locally" means CPU.

**This is the project's main risk.** It is why `BENCHMARK.md` exists and why it
must run before anything else is built.

**Also gained:** Whisper performs language identification as part of
transcription, so source-language detection costs nothing extra. It also emits
punctuation natively, which likely satisfies the bonus requirement's first
reading for free.

### D7. Google Cloud Translation for translation
500,000 characters/month, permanently free, and it does not expire like trial
credit. At roughly 900 characters per minute of speech, that is about **9 hours
of speech per month** — ample for development, rehearsal, and the demo.

**Why not the alternatives:** DeepL's Free plan can no longer be purchased by
new customers. Gemini's free tier is bounded by *requests* (15/min, 1000/day on
Flash-Lite), and at one request per utterance that caps usage at roughly 80
minutes of meeting per day — a real constraint at speech rates. Google Cloud
Translation has no comparable per-request ceiling.

**Noted:** Gemini's free tier may use submitted data for product improvement.
Relevant when the payload is meeting transcripts.

### D8. Free tiers only
User decision, reaffirmed after being shown that paid translation would cost
roughly 20–30 cents per hour of meeting. Recorded so it is not re-litigated.

### D9. Speaker tagging descoped to a late stretch goal
Not obtainable from loopback audio — the identity is gone before capture.

Routes that would provide it, and their costs: a diarization model gives
anonymous labels only ("Speaker 1"), never names, and adds load to the CPU that
is already the bottleneck. Real names require either scraping the meeting UI
(fragile) or joining as a bot participant (see D10).

**Decision:** ship without it. If the benchmark shows CPU headroom, add
anonymous diarization. Otherwise document *why* it is unobtainable — that
explanation demonstrates more understanding than a half-working diarizer.

### D10. No per-platform meeting integrations — rejected with evidence
- **Teams**: the real-time media platform is documented for **C#/.NET only**
  (Graph Communications SDK) and requires deployment on Windows Server. Wrong
  language, wrong runtime, wrong infrastructure.
- **Zoom**: RTMS provides live audio without a bot participant and is the
  closest fit, but requires a developer account plus approval, and covers Zoom
  alone. Zoom's own docs also confirm raw audio is available only via the
  desktop SDKs, not the web SDK.
- **Slack**: not researched. Do not assume either way.

**Rationale:** these are distribution mechanisms, not capability gains. The one
capability they would add — speaker identity — is already descoped by D9. Three
sets of accounts, approvals, and platform-specific code is not a
deadline-sized decision.

### D11. Final-only caption display (no interim rewriting)
Streaming ASR normally emits provisional text that gets revised as more audio
arrives. Showing that means captions visibly rewrite themselves, which reads as
broken.

**Decided by hardware:** two-tier display requires running Whisper repeatedly on
a growing buffer — several passes per utterance instead of one — on four
throttling cores that are already the bottleneck. Spending 2–3× the CPU on the
scarcest resource to produce a flickering effect is a bad trade.

**Mitigation for the resulting silence:** a live speaking indicator driven by the
VAD signal we already compute. Costs nothing, and converts "is this frozen?"
into "it's listening."

### D12. Segmentation: silence threshold OR max-length cap, whichever fires first
Pause-only segmentation has a failure mode: a speaker who talks for ninety
seconds without pausing produces nothing on screen for ninety seconds. A maximum
utterance length (start around 6–8s) bounds worst-case latency by construction.

**The tuning tradeoff, stated so it is not rediscovered:** a short silence
threshold gives faster output but cuts mid-sentence, which damages both
punctuation and translation quality — a translator handed half a sentence cannot
know what the sentence is. A short max-length bounds latency but causes more
mid-sentence cuts. These two numbers are where tuning time goes.

### D13. Local Python server + browser UI; desktop shell deferred
Fastest route to working software, and identical to whatever a desktop shell
would eventually wrap. Packaging does not affect the pipeline, so it can be
decided late at no cost.

Works for the demo as-is: the meeting runs in its desktop client, the UI is a
separate browser window, and capture happens at the OS layer between them.

### D14. Scaling explicitly out of scope
User decision. Recorded because the design would differ: at scale, transcription
would move server-side onto GPUs or a cloud ASR vendor billed per minute, since
GPU cost scales linearly with concurrent users. The current client-side design
is correct for a demo and wrong for a product — a distinction worth stating in
the journey doc, but not worth building for.

### D15. Transcription runs in a separate process from capture
Python's GIL means a CPU-bound Whisper call in a *thread* stalls the capture
loop and drops audio frames. Separate process, queue between them. This is not
an optimisation; it is a correctness requirement.

---

## 2026-08-18 — Session 2 (platform pivot)

### D16. Target platform moves to the Windows machine (GTX 1650 Ti, 4 GB)
The user has a Windows machine with an NVIDIA GTX 1650 Ti (4 GB GDDR6) and will
build and demo there. Transcription moves from CPU to CUDA.

**Why this is not a small change:** D6 recorded "the machine has no NVIDIA GPU,
so *locally* means CPU," and that single fact drove D9, D11, and the existence of
`BENCHMARK.md`. Every one of those is re-examined below rather than inherited.

**D6 is deliberately not edited.** It is an accurate record of what was true and
why the decision was made. Rewriting history to match the present would destroy
the thing this file exists for.

### D17. Linux descoped to the `AudioSource` interface only
`PipeWireMonitorSource` will not be built. D2's verification of the PipeWire
monitor path stays on the record as history — it was real, it just isn't the
target any more.

**Why:** one platform tested properly beats two tested half-way before a
deadline. The `AudioSource` Protocol (D4) already makes Linux a later addition
rather than a rewrite, which is exactly the value the abstraction was bought for.

**Cost, stated plainly:** the PRD's "runs on Linux and Windows, verified on both"
line is now false and has been changed rather than left aspirational.

### D18. VRAM replaces CPU throughput as the binding constraint
4 GB is the ceiling, and the budget is **not** 4 GB: the Windows desktop
compositor and the browser rendering our own caption UI both draw from the same
pool. The benchmark must therefore be run with a browser open, because the demo
has one.

The failure mode changes with it — from "falls progressively further behind" to
"CUDA OOM at model load or on a long utterance." The second is more abrupt and
easier to miss in a short test.

**Two properties of this specific GPU, to be measured rather than assumed:**
TU117 is compute capability 7.5, so CTranslate2 supports both `float16` and
`int8_float16`. But the GTX 16-series has **no tensor cores**, so "int8 is
faster" — reliable on RTX parts — is not a given here.

### D19. Distil-Whisper excluded from the model shortlist
`distil-large-v3` and its successors are **English-only**. The entire product
premise is transcribing a language the user does not speak, so an English-only
model cannot serve the source side no matter how fast it is.

**Recorded because it is a trap, not an oversight:** distil-whisper is the first
suggestion anyone (human or LLM) makes when asked to speed up Whisper, and the
benchmark numbers would look excellent right up until the demo language stopped
being English. `large-v3-turbo` is the multilingual answer to the same question.

### D20. `BENCHMARK.md` demoted from blocking gate to model-selection task
`STATE.md` said, in bold, "do not build the pipeline before this number exists."
That was correct when the question was **existential**: if Whisper could not keep
up on a CPU, the architecture had to change, and building on top of an unproven
assumption would have wasted the work.

CUDA answers the existential question. Some model will clear real-time on a
1650 Ti. What remains — *which* model, at which `compute_type` — is a value read
from config at startup, not an architectural fork. So it no longer gates code
that is independent of the model: audio capture and segmentation.

**This is recorded rather than quietly ignored** because a doc that says "don't
build yet" should be overruled explicitly, with the reasoning, or not at all.

**Still true, still required:** the benchmark runs before the Transcriber is
wired, and its sustained-throttling method stays — a laptop GPU has power and
thermal limits just as the U-series CPU did.

### D21. D9 and D11 stay closed until the benchmark says otherwise
Both were decided on hardware grounds, so a hardware change is a legitimate
reason to revisit them — but only with numbers. A 4 GB laptop part without tensor
cores is a modest GPU, and diarization would want VRAM that the caption UI is
already competing for. Reopening them on the general optimism that "we have a
GPU now" would repeat exactly the assume-instead-of-measure mistake the
benchmark exists to prevent.

### D22. Development on Linux, testing on Windows, git as the bridge
The user writes code on the Linux laptop and runs it on the Windows machine.

**Consequences that bind the code, not just the workflow:**
- Platform-specific dependencies carry environment markers
  (`PyAudioWPatch; sys_platform == "win32"`), so one `requirements.txt` installs
  on both machines.
- Windows-only imports are **lazy**, inside the function that needs them. If
  importing the package fails on Linux, the entire dev loop becomes
  push-pull-run for every typo.
- `WavFileSource` is the local smoke path, which is a second reason for it to
  exist beyond the deterministic-testing one in D4.

### D23. VAD is Silero, so the frame size is 512 samples / 32 ms
`INTERFACES.md` required picking the VAD first and then fixing the frame size to
match; this closes that. 16 kHz mono int16, 512-sample frames, never varied.

**Why Silero over WebRTC VAD:** the input is a whole desktop's audio output —
music, notification sounds, video stings, keyboard clicks. WebRTC VAD is an
energy-and-spectrum heuristic and fires readily on all of it; Silero is a trained
model and discriminates speech from non-speech far better. False positives here
are not cosmetic: each one is a wasted Whisper call on the GPU that is the
scarce resource (D18), plus a hallucinated caption on screen.

**Cost:** an ONNX runtime dependency and roughly 1 ms of CPU per 32 ms frame.
Cheap, and it is CPU work that no longer competes with transcription now that
transcription is on the GPU.

### D24. The source layer normalises audio; nothing downstream resamples
WASAPI loopback delivers the output device's **native** format — typically
48 kHz stereo, sometimes float32. Whisper and Silero both want 16 kHz mono
int16.

Downmix and resample happen **inside** `WasapiLoopbackSource`, so the
`AudioSource` contract in `INTERFACES.md` is literally true: every implementation
yields identical frames and no downstream stage contains a sample-rate branch.

**Why it is worth stating:** resampling in the wrong place is a classic source of
"it works with my test WAV but not live," because the WAV was already 16 kHz and
the live device never is.

---

## 2026-08-18 — Session 3 (system design requirements)

### D25. The project had a stability requirement, not a latency requirement
`BENCHMARK.md` gated on sustained **RTF < 0.5**. That gate guarantees the queue
does not diverge. It says nothing about how far behind the captions are, and the
DoD's "latency should be minimal" was never given a number — so a configuration
could pass every stated gate and still be unusable.

The arithmetic that was missing:

```
lag_after_speech_ends = silence_threshold + (RTF × D) + MT_roundtrip
word_age              = silence_threshold + MT_roundtrip + D × (1 + RTF)
```

`D` is the utterance duration, bounded by `max_utterance_ms`.

**The finding:** at the previous defaults (`max_utterance_ms` 6000–8000,
`silence_threshold_ms` 700, RTF 0.4, MT 300 ms), the first word of a caption
appears **12.2 s** after it was spoken. Every existing gate passes. That is a
subtitled recording, not a meeting anyone can participate in.

**Two consequences that are not obvious from RTF alone:**
1. `max_utterance_ms` is the dominant latency knob, not RTF.
2. RTF pays **twice** — it multiplies `D`. So the benchmark should select the
   *fastest model that is accurate enough*, not the largest that fits in VRAM.
   That is a different selection rule than `BENCHMARK.md` was written around,
   and it is amended there (D26's gate).

**Target chosen: p95 word age < 7 s.** Solving for `D` at the worst permitted
RTF, with `silence = 600 ms` and `MT = 300 ms`:

```
D ≤ (7.0 − 0.6 − 0.3) / (1 + 0.5) = 4.07 s
```

| | RTF 0.4 | RTF 0.5 (the gate) |
|---|---|---|
| lag after speech ends | 2.5 s | 2.9 s |
| word age | 6.5 s | 6.9 s |

**Derived config — these values are now derived, not placeholder ranges:**

| Key | Value | Supersedes |
|---|---|---|
| `silence_threshold_ms` | 600 | D12's "start around 500–800" |
| `max_utterance_ms` | 4000 | **D12's "start around 6000–8000"** |
| `min_utterance_ms` | 300 | new (D30) |
| `max_utterance_floor_ms` | 2000 | new (D28) |
| `asr_queue_maxsize` | 4 | new (D28) |

**D12 is not edited.** Its reasoning — that pause-only segmentation has an
unbounded worst case, and that the two numbers are where tuning time goes —
remains correct and is the reason a cap exists at all. What changed is that the
cap now has a derivation instead of an intuition. This is the same practice D16
established for D6.

### D26. `BENCHMARK.md` gains a third gate: p95 word age < 7 s
Sustained RTF < 0.5 and the VRAM ceiling both stay. The third gate is what makes
D25's target testable rather than aspirational, and it is measured end to end at
`max_utterance_ms = 4000` — not inferred from RTF, because MT round-trip and
queue wait are real and are not in the RTF number.

**Why a third gate rather than tightening the RTF one:** they measure different
failures. RTF answers "does this diverge"; word age answers "is this usable".
A model can pass either while failing the other, and collapsing them into one
number would hide which one broke.

### D27. Exactly two processes; translation stays with the server, not the GPU
D15 requires transcription to run in a separate process. It does not say what
else moves, and the answer is **nothing**.

- **Main process** (asyncio): WebSocket server, capture thread, Silero VAD +
  Segmenter, sentence splitter, translation client, publisher.
- **Worker process**: `faster-whisper` only. Owns the CUDA context and nothing
  else.

**Why VAD stays in the main process:** Silero costs ~1 ms per 32 ms frame (~3% of
one core, D23) and PyAudio's blocking read releases the GIL. The GIL problem D15
identified is specific to a multi-second CPU-bound Whisper call; a 1 ms model
does not reproduce it. A third process would buy nothing and add an IPC hop to
the latency budget D25 just defined.

**Why translation does not move to the worker:** it is network-bound. Putting it
behind the GPU would idle the GPU for the duration of every round trip — the
scarcest resource waiting on the least scarce one.

**Windows consequence that binds the code:** Windows uses `spawn`, not `fork`
(D22 covers the dev/test split; this is the runtime half). The worker re-imports
the package and loads the model cold — roughly 5–15 s, and the first run
downloads weights. This needs an explicit **ready handshake** before the UI
reports "running", or pressing Start looks like a hang. Designed for now rather
than discovered on the demo machine.

### D28. Backpressure: adaptive shrink, then drop the oldest
Bounded `Segmenter → Transcriber` queue, `maxsize = 4` (≈16 s of backlog — past
the D25 budget already, so it is a ceiling, not an operating point).

Policy, in order:

1. **Shrink.** `queue_depth ≥ 2` for 3 consecutive utterances → cut the effective
   `max_utterance_ms` by 25%, floor 2000. Shorter utterances cost less ASR each
   and drain the queue.
2. **Recover.** `queue_depth == 0` and RTF EMA < 0.35 for 30 s → grow back 25%,
   ceiling = configured value. Hysteresis both ways so it cannot flap.
3. **Drop.** Queue full → drop the **oldest** pending utterance, emit a `dropped`
   event.
4. **Signal.** Any shrink or drop puts the UI into a visible degraded state.

**Why drop the oldest rather than the newest:** a stale caption has no value. The
user is trying to follow a live conversation; handing them what was said 20 s ago
while the speaker has moved on is worse than a visible gap.

**Why shrink before dropping:** shrinking degrades translation quality slightly
(more mid-sentence cuts, the D12 tradeoff) but loses nothing. Dropping loses
content. Try the reversible degradation first.

**Why an unbounded queue was rejected:** it converts a transient GPU hiccup into
permanently growing lag with no recovery path, and it does so silently. The
current effective `max_utterance_ms` is published in the `health` message so the
adaptation is observable rather than mysterious.

### D29. The caption unit is the sentence, with fragment carry-over
D12 accepted that `closed_by = "max_length"` cuts mid-sentence, and that a
translator handed half a sentence cannot know what the sentence is. With
`max_utterance_ms` now 4000 (D25), max-length closes are *more* frequent, so
that accepted cost got larger and is worth paying down.

- Split the transcript on sentence boundaries.
- Complete sentences are translated and published **immediately**.
- The trailing fragment is held and prepended to the next utterance's text
  before splitting again.

**Why this is free:** only the fragment waits. Complete sentences are not delayed
at all, so it improves MT quality without spending any of the D25 budget.

**Required detail — the flush rule.** A held fragment must be flushed when
silence exceeds `2 × silence_threshold_ms`, on `stop`, or at session end.
Without it, the last words spoken in a meeting are held forever and never
appear. This is the obvious bug in the design and it is written down so it is
built, not found.

`preceding_silence_ms` attaches to the **first** sentence derived from an
utterance; later sentences carry 0.

### D30. Hallucination guard, because Silero reduces the input but does not clean it
Whisper emits confident garbage on near-silence — "Thank you.", subtitle-credit
boilerplate. D23 chose Silero partly to cut these, and it does cut them, but a
400 ms cough or a notification chime still gets through and still produces a
caption on screen that nobody said.

Reject a transcript when **any** of:
- the utterance is shorter than `min_utterance_ms` (300),
- `no_speech_prob` is above threshold,
- `avg_logprob` is below threshold.

**Contract consequence:** `Transcript` must carry `no_speech_prob` and
`avg_logprob`. `faster-whisper` returns both per segment at no cost, but they
have to be plumbed through, so they belong in `INTERFACES.md` rather than being
read off the model object at the point of use.

**Why this is a correctness requirement, not polish:** a fabricated caption is
worse than a missing one. The user cannot tell it is fabricated — that is the
whole point of the product — and they have no way to check.

### D31. The translation budget is persisted, cached, and skippable
D7 bought 500 000 characters/month, permanently free, not rolling over. Three
requirements follow that an in-memory counter does not meet:

- **Persist the counter** to a file keyed by month. Warn at 80%, hard-stop at a
  configurable ceiling. A process restart must not reset it — the realistic way
  to burn the demo budget is a testing loop across many runs, and an in-memory
  counter cannot see that.
- **LRU cache** on `(source_lang, target_lang, text)`. Short utterances repeat
  constantly in meetings ("okay", "yes", "thank you", "can you hear me"). Free
  characters and free latency.
- **Skip translation entirely when source == target.** Saves a round trip and
  the characters, and it is the common case the moment the user picks English
  while an English speaker is talking.

**Also:** a translation failure must never block the pipeline. Retry with
backoff, then publish the caption with `target_text = null` and an error flag.
The UI falls back to source-only — which is the reason source text is on the
card at all (D33).

### D32. Language identification accumulates over a window, then locks
`INTERFACES.md` said language is "detected on the first utterance… then locked".
With `max_utterance_ms` at 4000 (D25) the first utterance can be 800 ms of
"okay, hi" — and locking the whole session's language on that is a coin flip.

Accumulate Whisper's LID over the first **~10 s of speech** (not wall clock —
silence must not count), weight each vote by its confidence, then lock. Manual
override stays available at any time, as the PRD already requires.

**Why not just never lock:** re-running LID per utterance costs real GPU time,
and worse, it makes the source language flicker mid-meeting on a short or noisy
utterance, which changes the translation direction and produces visible
nonsense.

### D33. Caption cards show translation primary, source secondary
Target text large, source text small beneath it, toggleable.

**Why carry the source at all when the brief asks for translated text:** it is
the only way the user can tell an ASR error from an MT error. When a caption
reads oddly, "the model misheard the word" and "the model mistranslated the
word" call for different reactions, and without the source both look like the
app being wrong. It also gives D31's translation-failure path somewhere to
degrade to instead of a blank card, and it serves the partially-bilingual user,
who is a plausible real user of this product.

### D34. One JSONL timing log per session
One record per utterance: `t_speech_end, t_queued, t_asr_start, t_asr_end,
t_mt_end, t_sent, duration_ms, queue_depth, rtf, chars`.

**Why it is worth a decision entry:** this single file is the latency histogram,
the RTF curve, the throttling curve and the queue-depth graph — i.e. the entire
evidence base for the DoD's real-time claim and for the D25/D26 gates. It costs
roughly twenty lines. Building it after the fact means re-running every
measurement to get the numbers the journey doc needs.

---

## 2026-08-18 — Session 4 (stack confirmation and portability)

### D35. Silero runs from a vendored ONNX file; PyTorch is not a dependency
The `silero-vad` PyPI package declares `torch` and `torchaudio` as hard
requirements and does `import torch` at module scope in `model.py` — including
on the `onnx=True` path. Verified by inspection, and `import silero_vad` fails
in the dev venv today for exactly this reason.

**Why this is a portability decision and not a packaging preference:** on Windows
`pip install silero-vad` pulls torch (~2.5 GB, possibly a CUDA build). A torch
CUDA runtime in the same process as CTranslate2's CUDA runtime means two CUDA
contexts and two sets of cuDNN expectations on a card with **4 GB total**
(D18). It would work on the Linux dev box, where nothing loads CUDA, and fail on
the Windows machine as an OOM or a DLL error — the exact failure mode that is
expensive to diagnose remotely.

**Decision:** vendor `silero_vad.onnx` (2.3 MB, MIT) into the repo and run it
through `onnxruntime` directly. Never import the `silero_vad` package at
runtime. This also removes a first-run download.

Model interface, confirmed against the file:

```
inputs : input [N, 512] float32 · state [2, N, 128] float32 · sr int64
outputs: output [N, 1] float32  · stateN
```

**Measured, not estimated: 0.156 ms per 32 ms frame** on the dev laptop CPU —
about 0.5% of one core. D23 estimated "roughly 1 ms"; the real figure is ~6×
cheaper. D23 is left as written; this is the measurement that supersedes its
estimate.

### D36. The demo machine is the only machine that has to run this
Confirmed with the user. The brief asks for a UI "that you can demonstrate", not
one the interviewers install. No CPU fallback path will be built.

**What this buys:** the model shortlist stays CUDA-only, `BENCHMARK.md` keeps one
set of columns, and packaging stays a `uv sync`.

**What it costs, stated so it is a choice and not an oversight:** if an
interviewer asks to run it on their own laptop, the answer is no. The mitigation
is a recorded demo video plus the JSONL timing log (D34) as evidence, which is
stronger than a live install on unknown hardware anyway. Revisit only if the
truncated brief line turns out to demand distribution.

### D37. Google Cloud Translation over REST with an API key, not the client library
`google-cloud-translate` pulls gRPC and protobuf and expects a service-account
JSON file discovered through `GOOGLE_APPLICATION_CREDENTIALS` — an absolute path
that differs on every machine, which is a portability failure waiting to happen
across the Linux/Windows split (D22).

**Decision:** one `httpx` POST to `translation.googleapis.com/language/translate/v2`
with the key in an environment variable. Moves between machines by copying one
line into `.env`. The key is restricted to the Translation API in the console.

**Why this is not a shortcut:** the free-tier quota (D7) is per project, not per
auth method, so nothing about the budget changes. The Translator stays behind the
interface `INTERFACES.md` already requires, so swapping to the client library
later is a single class.

**Security note:** an API key is a bearer credential. It goes in `.env`, `.env` is
git-ignored, and `.env.example` carries the variable name only.

### D38. FastAPI + uvicorn; the UI is vanilla HTML/CSS/JS with no build step
One process serves the static UI and the WebSocket, and it is async-native, which
is where D27 already puts the segmenter, translator and publisher.

**No npm, no bundler, no framework.** A node toolchain would add a second
runtime, a second lockfile and a build step between a code change and a demo, in
exchange for nothing a caption list needs. The UI is one HTML file, one CSS file
and one JS file, opened directly from the server.

### D39. Dependencies pinned with `uv` and a committed cross-platform lockfile
`uv.lock` resolves for Windows and Linux in one file, including the
`sys_platform == "win32"` markers D22 requires, and is committed to git. The
Windows machine runs `uv sync` and gets byte-identical versions.

**Why not the `requirements.txt` D22 assumed:** a flat requirements file pins
direct dependencies but not the resolved transitive graph, so two machines
resolving on different days can land on different sub-dependency versions. Given
the whole workflow is "write on Linux, run on Windows" (D22), the lockfile is
the mechanism that makes "it worked on my machine" a testable claim rather than
a hope. `uv 0.11.7` is already installed on the dev machine.

**Consequence:** `pyproject.toml` + `uv.lock` replace `requirements.txt`. D22's
requirement that platform-specific deps carry environment markers is unchanged —
it just moves into `pyproject.toml`.

### D40. A `doctor` preflight command, built in step 1
`python -m speech_translator.doctor` checks and prints pass/fail for: Python
version, resolved package versions against the lock, `ctranslate2.get_cuda_device_count()`,
cuDNN/cuBLAS DLL load, Whisper model cache presence, a WASAPI loopback device,
and one 5-character live translation.

**Why it earns its place before the pipeline exists:** every item on the D41
register fails on the *other* machine, hours after the code was written, with an
error message pointing at the wrong layer. A preflight converts each of those
into a line item that says which thing is missing. It is also the first thing to
run on the Windows box after every `git pull`, which is the actual dev loop
(D22).

### D41. Portability register — the known "works on my machine" failures
Recorded as a checklist because each one is invisible on the machine that writes
the code.

| Risk | Mitigation |
|---|---|
| torch pulled in by Silero | eliminated by D35 |
| cuBLAS/cuDNN DLLs missing for CTranslate2 on Windows | take them from the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` **pip wheels**, pinned in the lockfile — not hand-copied DLLs, which are unreproducible |
| Whisper weights downloaded on first run | pre-download; pin the cache directory in config; surface it in the `loading` state (D27) |
| Google API key absent | `.env.example`; caught by `doctor` (D40) |
| **Windows console is cp1252** | printing a Spanish or Japanese caption to the terminal raises `UnicodeEncodeError` and kills the thread it happens on. Force UTF-8 on all stream handlers and log files |
| Audio device variance | read the device's `defaultSampleRate`; never hard-code 48000; the source normalises (D24) |
| Default output device changes mid-session (headphones) | detect the stream error and surface `error` state with a re-select action rather than dying silently |
| CRLF/LF across two machines | `.gitattributes` with `* text=auto eol=lf` |
| Port already in use | port is config, not a literal |

---

## 2026-08-18 — Session 5 (Level 0, foundation)

### D42. UTF-8 is forced at package import, not at each entry point
D41 requires UTF-8 on every output stream and log file. The obvious
implementation is a call at the top of each `main()`, and the obvious failure is
the entry point that forgets — most likely the transcription worker, which on
Windows is `spawn`ed and re-imports the package as a fresh interpreter (D27).

`force_utf8()` therefore runs in `speech_translator/__init__.py`. Importing the
package is the one thing every process does, so the mitigation cannot be
skipped by a new entry point that nobody remembered to update.

**Two details that are choices, not accidents:**
- `errors="replace"`, not `strict`. A console that genuinely cannot render a
  glyph should print a replacement character; killing the capture thread over a
  font is a worse outcome than a mangled character.
- Reconfiguration failures are swallowed. Under pytest capture or a binary pipe
  there is nothing to reconfigure, and that is not an error condition.

The regression test reproduces the Windows console on Linux by forcing a
`cp1252` stdout in a subprocess and printing a Spanish and a Japanese string.
Without `force_utf8()` that child raises `UnicodeEncodeError`; with it, it
passes. The cp1252 risk is now tested on the machine that cannot reproduce it
naturally.

### D43. The CUDA pip wheels are necessary but not sufficient on Windows
D41 sources cuBLAS and cuDNN from `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
rather than hand-copied DLLs, because wheels are pinned by the lockfile. Found
while building `doctor`: installing them does not make them **loadable**. The
wheels unpack to `site-packages/nvidia/<component>/bin`, and nothing puts those
directories on the Windows DLL search path. PyTorch does this for its users;
CTranslate2 does not, and we deliberately have no torch (D35).

The symptom would have been `import ctranslate2` failing with a bare
"DLL load failed while importing translator", on the demo machine, pointing at
CTranslate2 rather than at the missing cuDNN — exactly the diagnosis-at-a-
distance D40 exists to prevent.

**Decision:** `speech_translator/windows_cuda.py` walks the installed `nvidia`
namespace package and calls `os.add_dll_directory()` for each `bin` directory
holding DLLs. No-op on Linux; safe to import anywhere (D22). It must be called
**before** the first `import ctranslate2` — in `doctor`, and in the worker
process at Level 4, which re-imports everything from scratch under `spawn`.

Recorded because the fix is invisible: if it is ever deleted, everything still
works on Linux and the failure appears only on the machine that matters.

### D44. `doctor` grades a failure by whether this platform can pass it
Half of `doctor`'s checks — CUDA, the cuDNN DLLs, WASAPI loopback — cannot pass
on the Linux dev box, by design (D17, D22). If those printed as ordinary
failures, the dev-box run would show a wall of red every time and stop being
read, which defeats the point of building it in step 1.

Each check is therefore tagged `windows_only`. Such a check still prints
**FAIL** on Linux — the honest answer, and the one asked for — but it is
labelled "expected here" and excluded from the exit code. So:

- Linux, healthy: red lines, exit 0.
- Windows, healthy: no red lines, exit 0.
- Either machine, genuinely broken: red lines and exit 1.

This keeps `doctor` usable as a CI-style gate on the demo machine while staying
truthful on the dev machine, instead of picking one at the cost of the other.

`doctor` also grades two things as **WARN**, not FAIL: an unselected
`model_size`/`compute_type` (correct until Level 3 runs — D20) and absent
Whisper weights (correct until the first run downloads them). A preflight that
cries wolf about the expected state of an unfinished project is a preflight
people stop running.

### D45. The lockfile is verified for both platforms at lock time
D39 committed `uv.lock` so both machines resolve identically. A lockfile
produced on Linux can still be missing Windows wheels for a package it happens
to resolve from source locally, and that would be discovered on the demo
machine.

Two things close it:
- `[tool.uv] required-environments` names both
  `linux/x86_64` and `win32/AMD64`, so `uv lock` fails rather than producing a
  lock that cannot install on Windows.
- `requires-python = ">=3.12,<3.13"`. An unbounded ceiling lets the Windows box
  resolve a different Python and therefore different wheels; `uv` will fetch a
  matching 3.12 there rather than "whatever is installed" deciding it.

Verified: all 45 locked packages carry a `win_amd64` wheel, a pure-Python wheel,
or an sdist — including `ctranslate2`, `onnxruntime`, `av`, `tokenizers`,
`soxr`, `pyaudiowpatch` and both `nvidia-*` packages. This is evidence, not
proof; the proof is `uv sync` on Windows, which is owed.

---

## 2026-08-18 — Session 6 (Level 1, audio capture)

### D46. One `soxr.ResampleStream` per source, and the 16 kHz path bypasses it
D24 said the source normalises. It did not say *how*, and the obvious-looking
implementation is wrong: `soxr.resample(chunk, 48000, 16000)` per chunk. A
polyphase resampler is stateful — it has a delay line — and rebuilding it per
chunk discards that line and welds together fragments that do not line up. The
result is a click at every chunk boundary, which is `PLAN.md`'s stated risk for
this level.

**Decision:** `FrameFormatter` constructs exactly one `soxr.ResampleStream` in
`__init__` and never rebuilds it. `flush()` drains its delay line with
`last=True`.

**Measured on the dev box** — the same 10 s two-tone signal, pushed in random
1–5000-sample chunks:

| | out-of-band energy | max sample step | output samples (expect 160 000) |
|---|---|---|---|
| One stream (what we ship) | **−81.4 dB** | 8 584 | 160 000 |
| A fresh stream per chunk | **−30.2 dB** | 11 218 | 159 994 |

51 dB of broadband noise is what "clicks at chunk boundaries" is, numerically.

**Second half of the decision:** when the source is already 16 kHz mono int16 —
which every recording *we* produce is — no resampler is constructed at all, no
downmix runs, and no float round-trip happens. `passthrough` is True and the
bytes reach the framer untouched. Bit-exactness is then a structural property,
not a rounding claim, which is why the test can assert byte equality rather than
a tolerance.

**The guard is mutation-tested, not just asserted.** `tests/test_audio.py` has
`test_control_a_per_chunk_resampler_does_produce_seams`, which deliberately
reproduces the bug and asserts the seam bound *rejects* it — otherwise a bound
loose enough to accept anything would pass silently. Injecting the per-chunk
rebuild into `FrameFormatter` fails two tests (the output guard and the
constructed-once guard) and no others; reverting it restores 50/50.

### D47. `record_loopback` writes the normalised stream, not the device stream
The tool could write what the device gave it (48 kHz stereo, possibly float32)
or what the pipeline consumes (16 kHz mono int16). It writes the second: the
exact bytes `AudioSource.frames()` yields.

**Why, in order of weight:**
1. `BENCHMARK.md` step 2 feeds these recordings to faster-whisper at Level 3.
   Whisper wants 16 kHz mono; writing the device format would mean Level 3
   re-implements the conversion, which is the sample-rate branch D24 exists to
   prevent.
2. Replaying our own capture then goes through the passthrough path, so an
   offline run is bit-identical to the live one rather than resampled twice.
3. **The file becomes the evidence for the format layer.** If it plays back at
   the right pitch with no clicks, decode → downmix → streaming resample →
   framing is correct end to end. A device-native recording would be a recording
   *of the device* and would prove nothing about our code.

**Rejected:** an `--also-raw` flag writing a second device-native WAV as a
diagnostic. **Cost, stated plainly:** if the Windows recording sounds wrong there
is no raw capture to separate a device problem from a resampler problem. Accepted
because the resampler side is already covered by D46's measurement, so suspicion
falls on capture first and the diagnosis is not actually ambiguous.

### D48. `--input-wav` makes the tool itself testable on the dev box
`record_loopback` is more than its capture call: incremental WAV writing, the
level meter, the Ctrl-C path that must still leave a valid file, the duration
report. On Linux none of that would ever run, and it would arrive on the demo
machine untested at the same moment as the device code (D22).

`--input-wav PATH` substitutes `WavFileSource` for the live source through the
same `open_source()` seam. Everything except opening the device now runs, and is
tested, here. What remains Windows-only is genuinely Windows-only.

### D49. WASAPI: ask for int16, accept float32, read the rate off the device
Two things the demo machine decides for us, and both are on the D41 register.

- **Sample rate.** `defaultSampleRate` is read from the device info and handed to
  the formatter. 48 000 is typical, 44 100 and 96 000 are not exotic, and the
  user can change it in Windows sound settings between two runs. The literal
  48000 appears nowhere in the audio package.
- **Sample format.** We request `paInt16` first because it is what the pipeline
  wants and it skips a conversion. WASAPI shared mode may insist on the device
  mix format, commonly float32, so `_open()` falls back to `paFloat32` and
  records which one it got. Both feed the same `FrameFormatter`, so D24 holds
  either way; the float path clips to ±1.0 *before* scaling, because a mixed
  desktop can exceed unity and a wrapped sample is a loud click where a clipped
  one is not.

Which format the GTX machine actually negotiates is unknown until it runs there,
and is written down as an open item rather than assumed.

### D50. Blocking `stream.read`, not a callback and a queue
PortAudio offers both. The callback would never block the audio thread and would
make overruns countable; the blocking read is one loop with no second
concurrency surface.

**Blocking read wins for now** because PortAudio already buffers internally, the
only downstream consumer is Silero at 0.15 ms per 32 ms frame (D35) — a 200×
margin — and D28 already owns backpressure further down the pipeline. Adding a
second, differently-shaped drop policy at the capture end would mean two things
that can drop audio for unrelated reasons.

**The condition for revisiting is explicit:** if the ten-minute Windows recording
shows overruns or gaps, swap in a callback plus a bounded queue. The change is
local to `wasapi.py` because the formatter and the `AudioSource` contract do not
move.

---

## 2026-08-18 — Session 6 (Level 3 preparation)

### D51. The benchmark gets a harness, and gate 3 is predicted at 3, measured at 7
Two inconsistencies in `BENCHMARK.md`, both found by asking what Level 3 would
actually *do* on the day rather than what it says.

**The harness.** `BENCHMARK.md` describes a method — 7 configurations, 10
sustained minutes each, per-chunk timing, first-minute vs last-minute RTF, VRAM
polled across the run — and `PLAN.md` said "not code, a measurement". Both are
true of the *deliverable* and neither is true of the doing: 70 minutes of runs
with that much bookkeeping is not something anyone drives from a REPL, and a
mis-typed loop discovered at minute 55 costs the whole session.

`tools/benchmark.py` is therefore a Level 3 deliverable. It is **written and
dry-run on Linux** against `WavFileSource` with a tiny model on CPU, and
**executed on Windows**. The only Windows-specific thing about it is that a GPU
answers; every loop, timer and table row is platform-neutral. This is the same
split as D48 — shrink "Windows-only" to the part that genuinely is, so the first
run on the demo machine is a *run* and not a debugging session.

**Gate 3.** `BENCHMARK.md` said "measure p95 word age end to end, not by plugging
RTF into the formula". At Level 3 there is no end to end — no Transcriber wired,
no Translator, no UI; those are Levels 4, 5 and 6. The instruction was written
when the benchmark was step 1 and was not revisited when D20 moved it.

Reconciled rather than deleted, because the underlying point stands:

- **Level 3 predicts it**, from measured RTF plus a **measured** MT round trip —
  and that second term is the part worth insisting on. It is a live API call with
  a real latency; substituting a guessed 300 ms into a gate is exactly what D25
  exists to stop. Time ten calls, take the p95. The prediction's job is to *rank
  seven configurations*, and it is sufficient for that.
- **Level 7 measures it**, end to end, off the JSONL log (D34), where queue wait
  and the browser competing for the GPU are visible.

The results column is labelled *predicted* so the two are never confused. A
prediction reported as a measurement is the failure mode; a prediction reported
as a prediction is just the right tool for choosing a model.

### D52. Level 3 may run before Level 2, at a stated cost
`PLAN.md`'s rule is one level at a time. Level 3 is the exception worth naming,
because its dependency on Level 2 is weaker than the ladder implies: the
benchmark chunks audio at a **fixed** 4 s (D25), which needs no Segmenter.

What Level 2 actually contributes is a *realistic chunk-length distribution*.
Real utterances close on silence far more often than on max-length, so the true
distribution skews shorter than 4 s — and shorter chunks amortise per-call
overhead over less audio, which makes RTF **worse**. A fixed-4 s benchmark is
therefore optimistic, not neutral.

**If Level 3 runs first, the mitigation is to bracket rather than assume:** run
the selected configuration at 1.5 s as well as 4 s and report both. If the short
bracket fails a gate the 4 s number passes, that is the finding — and it is one
the headline table would otherwise have hidden until Level 7.

Recorded because it is a deliberate deviation from a stated project rule. The
argument for taking it is access: Levels 0 and 1 already owe the Windows machine
a `uv sync`, a `doctor` run and a 10-minute capture, and the benchmark needs that
capture as its input, so all four fit in one session on that machine. The
argument against is that the headline RTF is then approximate until Level 7
confirms it.

---

## 2026-08-18 — Session 7 (Level 2, segmentation)

### D53. Silero v5 needs 64 samples of context; the 512-wide call was silently dead
`INTERFACES.md` §2, D35, `assets/README.md` and `doctor` all documented the
model's input as `[N, 512]`. The vendored file is Silero **v5**, whose 16 kHz
call takes **576** samples: the 512 new ones with **64 samples of the previous
frame prepended**. The reference implementation keeps that context in the wrapper
alongside the LSTM state; ours had no wrapper yet, so it kept neither.

The ONNX input dimension is dynamic. A 512-wide call therefore *runs*, returns a
float, and raises nothing. Measured on real recorded speech
(`/usr/share/sounds/alsa/Front_Center.wav`, peak 15 211 — a 1.4 s spoken clip):

| call shape | silence | 440 Hz tone | music chord | white noise | **real speech** |
|---|---|---|---|---|---|
| 512, as documented | 0.001 | 0.001 | 0.001 | 0.002 | **0.0005** |
| 576 = 64 context + 512 new | 0.004 | 0.000 | 0.006 | 0.011 | **1.000** |

**So two things carry across a frame boundary, not one:** the LSTM `state`
`[2, 1, 128]`, and a 64-sample audio context. This is D46's failure one layer up
— state that must survive a boundary, discarded at the boundary — and it gets
D46's treatment: a guard for each, and a control for each proving the guard can
see the bug.

**What makes this worth a decision rather than a bugfix note** is how it would
have failed. A VAD that answers "no speech, ever" produces no utterances, so no
transcripts, so no captions, and every stage reports healthy. The symptom on the
demo machine is a blank screen with a green health badge, and the natural
suspects — capture, CUDA, the model download — are all downstream or upstream of
the actual fault.

**`doctor` is corrected too, and differently.** It used to construct the session,
push one frame through and report the timing. That is a check that the file
*loads*, dressed as a check that the VAD *works*, and it passed throughout. It
now runs the speech fixture (D56) and asserts speech > 0.9 and silence < 0.1. A
preflight that cannot fail on a dead component is not a preflight.

D35 is not edited: its reasoning about torch and its 0.156 ms budget both stand,
and the re-measurement at the correct 576 width is 0.116–0.141 ms, still inside
it. The signature block in D35 is superseded by this entry.

### D54. Speech threshold 0.50, release 0.35 — TUNABLE, and hysteresis rather than a single cut
Silero emits a probability. Nothing in D23 or D25 says where to cut it, so this
is a choice, and it is labelled in `config.py` the way D30's thresholds are:
**TUNABLE, not derived.**

**Two thresholds, not one.** Open at 0.50, stay open down to 0.35 — the same
0.15 gap Silero's own reference iterator uses. Measured falling edges on real
speech run `1.00 → 0.98 → 0.74 → 0.11`, and mid-word dips land in the 0.4s
(0.46 and 0.41 both observed); a single cut re-closes the utterance there, which
fragments an utterance mid-word and hands the translator half a clause (the D12
tradeoff, incurred for no reason).

**The flicker question, answered separately for the two things that flicker:**

1. *The segmenter* is stabilised by the hysteresis above.
2. *The `speaking` indicator* is stabilised by a **160 ms off-debounce**, which
   applies to the side channel only and never to segmentation.

**No minimum speech run to open, deliberately.** It is the obvious third
mechanism and it is the wrong one here: requiring N consecutive speech frames
delays the utterance's start, which spends latency budget (D25) to solve a
problem D30 already solves for free at the other end — a blip that opens an
utterance is discarded at close, having cost nothing but a few frames of
buffering. Adding a min-run would mean two mechanisms suppressing the same
false positive, one of them by making the product slower.

**160 rather than 96 is a measurement, not a preference.** At 96 ms the
indicator blinked dark for a single frame mid-phrase on the fixture: the natural
dip around a plosive runs to three sub-threshold frames, which is exactly where a
96 ms window expires. At 160 ms it holds through those and still goes dark at
real pauses — a 96 ms dark gap remains at the comma in the first phrase, which is
correct, because there *is* a pause there. The indicator resolves pauses down to
about 250 ms and holds through anything shorter.

### D55. Pre-roll 128 ms, tail pad 192 ms, and the min-utterance check measures speech
Silero reports on the frame in which speech crosses the threshold, which is at
best the frame containing the onset. Handing Whisper an utterance that begins
exactly there clips the leading phoneme, and a clipped first word is a
transcription error that no downstream stage can repair.

- **Pre-roll: 4 frames (128 ms)** kept from before the trigger, out of a ring
  buffer that is running anyway.
- **Tail pad: 6 frames (192 ms)** kept after the last speech frame; the rest of
  the 600 ms that closed the utterance is **dropped**. Trailing silence is
  exactly what D30 says Whisper hallucinates "Thank you." onto, and `end_ms`
  should mean the end of speech.

Both are TUNABLE, both are whole frames, and the tail pad is deliberately less
than `silence_threshold_ms` so it cannot eat the gap that closed the utterance.

**The consequence that is easy to miss, and was nearly shipped:** pre-roll plus
tail is **320 ms of padding**, which is more than `min_utterance_ms` (300). If
the D30 discard tested `end_ms - start_ms`, every blip would clear the bar and
the guard would be dead code — a 96 ms cough would emit a 416 ms "utterance". So
the discard measures **speech**, between the first and last speech frames. This
was caught by writing the blip test and getting an utterance back.

Every emitted utterance therefore contains at least `min_utterance_ms` of speech,
which is strictly stronger than the transcriber's own check in `INTERFACES.md`
§3, and Level 4 should know that rather than rediscover it.

**Two invariants fall out and are asserted for every utterance:**
`len(pcm) == (end_ms - start_ms) * 32`, and
`preceding_silence_ms == start_ms - previous_emitted.end_ms`.

The max-length cap also floors to a whole frame, so "no utterance exceeds
`max_utterance_ms`" holds for caps that are not multiples of 32 ms — at the
2000 ms backpressure floor it emits 1984 ms, not 2016.

### D56. Level 2 is tested with a scripted VAD and a generated speech fixture
Two problems, and they need different answers.

**The state machine** must not be tested through Silero. Boundary arithmetic
deserves exact inputs, so `Segmenter` takes a `SpeechDetector` and the tests pass
it a list of probabilities. Every closing rule, discard, gap and override
assertion is deterministic.

**The wrapper** must be tested through Silero, or D53 recurs. Its negative
controls are easy — silence, a 440 Hz tone, a four-note chord and white noise all
read below 0.05, and all produce zero utterances end to end, which is `PLAN.md`'s
own criterion. Its positive control is the problem: **nothing synthesisable from
numpy reliably crosses 0.5.** Formant-synthesised vowels with a syllable-rate
envelope peak around 0.8 on 6% of frames, which is not something to assert on.
Without a positive control, the entire stage can be dead and green.

**Decision: generate speech and commit it.** `tools/make_speech_fixture.py`
drives `libespeak-ng` through ctypes and writes `assets/speech_fixture.wav` —
13.92 s, three phrases separated by known 900–1000 ms silences, at 16 kHz mono
int16. Byte-reproducible across runs.

- **Generated rather than downloaded** because espeak-ng's default voice is
  formant synthesis: the output embeds no recorded audio and no third-party
  sample, so there is no licence to reason about. mbrola voices are recorded
  diphones and are deliberately not used.
- **Committed rather than generated at test time** because the alternative is
  `pytest.skip` on any machine without espeak-ng, which means skipping on
  Windows — the machine that matters.
- The third phrase is deliberately longer than `max_utterance_ms`, so the
  max-length path is exercised by real audio and not only by a fake VAD.

**Stated limit, because this is the tempting place to overclaim.** TTS speech has
no room tone, no music bed, no overlapping speakers and no reverb — precisely
what D23 chose Silero to survive. The fixture proves the VAD is alive and the
state machine behaves. It is **not** the captured meeting WAV `PLAN.md` asks for,
and Level 2's first acceptance criterion stays open until that recording exists.

---

## 2026-08-19 — Session 8 (Level 3, the benchmark harness)

### D57. The benchmark chunks at the Segmenter's real utterance distribution
`BENCHMARK.md` step 3 said "process the sample in chunks matching the planned
utterance length — **4 s, per D25**", with a clause beginning *"if Level 2 is not
built yet"*. Level 2 is built (649740a), so the clause no longer applies and
**D52's stated cost does not have to be paid**: a fixed-4 s benchmark is
optimistic, not neutral, because real utterances close on *silence* far more
often than on max-length and shorter chunks amortise per-call overhead over less
audio.

**Decision: the primary chunker is the real `Segmenter`, driven over the
recording.** Three reasons, in order of weight:

1. **It is the workload.** A fixed 4 s slice contains the closing silence the
   Segmenter would have trimmed and lacks the pre-roll it would have prepended
   (D55). Same audio file, different bytes, different cost. Measuring the slices
   measures something the app never sends.
2. **It removes the optimism** D52 had to accept on the way in. The distribution
   is measured and reported (p50 / p95 / max) rather than assumed, so if it comes
   out near 4 s after all, that is a finding rather than a coincidence.
3. **It is comparable across rows.** The Segmenter is deterministic over a WAV,
   so the harness cuts the audio **once, up front** and hands byte-identical
   chunks to all seven configurations. Otherwise an RTF difference between two
   rows could be a difference in what was measured. It also puts the VAD's own
   cost outside the timed region, where it belongs.

**D52's bracket is kept, not dropped.** `--chunking fixed --chunk-ms 4000` and
`--chunk-ms 1500` still exist, and are run for the **selected** configuration
only — about 20 minutes, rather than tripling a 70-minute matrix. If the short
bracket fails a gate the segmenter run passes, that is the finding, and it is the
one the headline table would otherwise hide.

`BENCHMARK.md` step 3 is rewritten to match. **D52 is not edited**: it is an
accurate record of the argument as it stood, and this project supersedes by
appending (D12 under D25, D6 under D16) rather than by rewriting history.

**Cost, stated:** the chunk list now depends on the VAD thresholds, which are
TUNABLE and not yet calibrated against real meeting audio (D54, D55). A retune at
Level 7 changes the distribution and therefore the RTF denominator. The mitigation
is that the distribution is recorded in `results.json`, so a later comparison is
possible rather than guesswork.

### D58. The Linux dry run gets both a stub engine and a real CPU model, and the stub is the default
D51 requires the harness to be dry-run on Linux before it is trusted on Windows.
Two ways to do that and they prove different things, so the answer is both — but
which one is the default matters.

- **`--engine stub`, the default for `--dry-run`.** No weights, no network, no
  GPU, and an injectable clock, so a simulated ten-minute run costs
  milliseconds. This is what lets the dry run live in `pytest` rather than be
  something a person remembers to do. It exercises the chunk loop, the warm-up
  exclusion, model-load exclusion, the wall-clock minute split, the percentiles,
  the gate arithmetic, the incremental writes, the OOM path and the table.
- **`--engine faster-whisper --device cpu`.** A real `WhisperModel`, real
  weights, real decoding. This is the once-per-change smoke test, and it is the
  only thing that proves the call signature is right and that the run is not
  timing an empty generator.

**Why not the stub alone:** `model.transcribe()` returns a **lazy generator** and
decodes nothing until it is drained. A harness that times the call without
consuming it reports an RTF near zero and is internally consistent while doing
so. A stub cannot catch that, because a stub does whatever the harness asks. So
the generator is drained inside `FasterWhisperEngine.transcribe`, and there is a
test with a fake `WhisperModel` returning a genuinely lazy generator that asserts
it was consumed — no weights required.

**Why not the real model alone:** it needs a network and ~75 MB on first run,
which is exactly the kind of dependency that turns a test suite into a thing
people skip.

**What the dry run does not touch, stated so it is not mistaken for coverage:**
`get_cuda_device_count() == 1` against a device, `add_cuda_dll_directories()`
against a real DLL path, VRAM polling (there is no `nvidia-smi` on the dev box),
thermal throttling, and a genuine CUDA OOM. Those are owed to the Windows
machine — which is the point of naming them here rather than letting a green test
run imply otherwise.

### D59. "Ten sustained minutes" is ten minutes of wall clock, and the audio loops
`BENCHMARK.md` says "run for at least 10 minutes continuously per
configuration"; D51 and `PLAN.md` both price the matrix at ~70 minutes. Those
only reconcile one way. Ten minutes of *audio* at RTF 0.3 is a three-minute run,
the whole matrix is ~25 minutes, and the GPU never gets hot enough for
"first-minute vs last-minute RTF" to measure anything — which is the number
`BENCHMARK.md` says predicts demo behaviour.

**Decision: chunks are fed back-to-back, looping the recording, until the
wall-clock budget is spent.** Laps and total audio processed are recorded, and
the first/last-minute windows split on wall clock. A ten-minute recording is
therefore played through roughly three times per configuration.

Looping is safe here because `condition_on_previous_text=False` (step 4) makes
each chunk independent — there is no cross-chunk state to be corrupted by a seam
back to the beginning.

**Stated cost.** Back-to-back is a *heavier* duty cycle than the live app, whose
GPU is idle between utterances for the fraction of time given by `1 − RTF`. So
this overstates thermal load. That is the direction to be conservative in for a
gate whose whole purpose is to catch a machine that flatters itself when cold,
and the alternative — pacing chunks at wall-clock speed — would spend ten minutes
to collect three minutes of GPU time.
