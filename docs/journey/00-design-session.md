# Journey 00 — Design session (2026-08-18)

Chat-only session with Claude (Opus 5) in Claude Code. No code written by
design; the goal was to pin down requirements and architecture before any
implementation session starts.

## How this session was run

I set ground rules up front: separate LLM sessions for separate roles
(read-only, code, edit, test, evaluate), frequent new sessions to control token
usage, and an explicit instruction to the model — **do not agree with me, point
out where I'm wrong, and don't state anything you aren't sure of or don't have
real numbers for.**

That instruction changed the session's output materially. Several times the
model told me something I proposed wouldn't work, rather than building it.

## What the LLM got right

**It refused to guess at my hardware.** Instead of estimating whether Whisper
would run, it asked permission and then actually inspected the machine —
`lscpu`, `free`, `lspci`, `nvidia-smi`. Finding no NVIDIA GPU reframed the whole
project: with no CUDA, transcription is CPU-bound, and that single fact drove
most of the decisions that followed.

**It verified the audio path rather than asserting it.** It claimed PipeWire
exposes a monitor source, then went and confirmed it — `port.monitor = "true"`
on sink node 47. When I later said I didn't want to be Linux-only, it searched
for and found `PyAudioWPatch` for Windows WASAPI loopback, with the wheel
support matrix, rather than telling me "Windows has an API for that."

**It corrected a wrong assumption of mine, twice.**
1. I asked for speaker tags — `Ravi: [text]`. It explained that loopback audio
   is a single mixed waveform: the meeting client discards speaker identity
   before the audio reaches the sound card, so nothing downstream can recover
   it. That reframed a "small UI addition" into an architectural decision.
2. I assumed speaker tagging made the app two-way. It separated the two axes:
   tagging is *who spoke* (still incoming), two-way is *direction* (needs my
   microphone). Ten tagged speakers would still be one-directional.

**It told me when I was off-brief.** I asked directly whether I was deviating.
It said yes — speaker tagging and multi-user scaling appear nowhere in my
assignment, which explicitly says "focus on the most important functionality."
It then drew a useful line: *thinking* about them belongs in this journey doc,
*building* them does not.

**It refused to give me numbers it couldn't source.** Asked about free tiers, it
searched rather than answering from memory, and it labelled confidence per
source — Google's own docs versus aggregator blogs. It also flagged that DeepL's
free plan is closed to new customers, which I'd have discovered the hard way.

**It found the fact that killed an idea.** I wanted a proper in-platform
integration. It researched and found that Teams' real-time media platform is
C#/.NET only and requires Windows Server deployment — a hard stop for a Python
project on a deadline. That is a better answer than a vague "that would be
complex."

## Where I had to push back

I twice asked whether we could stop discussing my hardware. The answer was that
free-tier-only forces transcription onto my laptop, and my laptop is the weakest
link — so the hardware *is* the constraint. Fair, but it took me asking directly
to get that explanation, rather than it being stated once and clearly up front.

The model also kept asking for the same two missing inputs (three truncated
lines in my brief, and my deadline) across five turns. Correct behaviour — they
genuinely gate scope — but it repeated the ask more than it needed to.

## Decisions this session produced

Recorded in full with reasoning in `DECISIONS.md`. The load-bearing ones:

- System audio capture, because it is the only approach agnostic to the meeting
  platform — one code path covers Zoom, Teams, Meet, Slack, and anything else
  that makes sound.
- Final-only captions rather than live-updating text, because two-tier display
  would cost 2–3× the CPU on my scarcest resource to produce visible flicker.
- Segmentation on silence **or** a max-length cap, because pause-only
  segmentation shows nothing for 90 seconds when someone monologues.
- Google Cloud Translation on the permanent free tier (~9 hours of speech per
  month) over Gemini, whose free tier is capped by *requests* in a way that
  bites at speech rates.

## What I learned

The most useful thing to come out of this session was a reframing. I had been
asking "how fast is Whisper on my machine," which is a tuning question. The real
question is whether transcription runs **faster than audio arrives** — because
if it doesn't, the delay compounds and the app falls further behind every
minute. That is binary, not tunable, and it means one benchmark can invalidate
the entire plan. It is now the first task of the next session.

## Next

Run `BENCHMARK.md`. Everything else waits on that number.

---

**Screenshots to capture next session:** benchmark output per model size, the
throttling curve from first to last minute, and `wpctl status` showing the
monitor source being captured.
