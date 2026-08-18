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
