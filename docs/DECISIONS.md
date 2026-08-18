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
