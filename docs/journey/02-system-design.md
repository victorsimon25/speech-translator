# Journey 02 — System design requirements: RTF is not latency (2026-08-18)

Session with Claude (Opus 5) in Claude Code, in plan mode. Chat only, by
intent — no production code. This session closed the system-level requirements
before the first stage got written, and found that the project's central
performance number was measuring the wrong thing.

## The prompt that started it

> go thrugh the docs and give me the status and things to proceed further

then, after the status report:

> let us discuss on the system design requirements for this project and this
> will also be a chat only session

## The finding

I expected a summary of what was already agreed. What came back instead was an
objection to a number that had survived two sessions unchallenged.

`BENCHMARK.md` gated on **sustained RTF < 0.5**, and `PRD.md` asked for "minimal
latency" without defining it. The model's point was that these are not the same
requirement and the first does not imply the second:

```
lag_after_speech_ends = silence_threshold + (RTF × D) + MT_roundtrip
word_age              = silence_threshold + MT_roundtrip + D × (1 + RTF)
```

Then it put the project's own defaults into that formula — `max_utterance_ms`
6000–8000 from D12, `silence_threshold_ms` 700, RTF 0.4, MT ~300 ms:

| | 8 s utterances | 4 s utterances |
|---|---|---|
| Caption appears after speaker stops | **4.2 s** | 2.6 s |
| Age of the *first word* in that caption | **12.2 s** | 6.6 s |

Twelve seconds. Passing every gate the project had written down. I would have
built it, measured RTF at 0.4, ticked "real-time" on the Definition of Done, and
found out at the demo.

Two things follow that are not obvious from RTF alone:

1. **`max_utterance_ms` is the dominant latency knob**, not RTF. D12 picked
   6–8 s from segmentation reasoning — how much context a translator needs — and
   nobody had checked what it did to the delay.
2. **RTF pays twice**, because it multiplies `D`. So the benchmark's selection
   rule was wrong too: it was set up to find the largest model that fits in 4 GB,
   when it should find the *fastest model that is accurate enough*.

## My assessment of the output

This is the most useful thing an LLM has done on this project, and it is worth
being precise about why. It did not know anything I could not have worked out.
The arithmetic is one line. What it did was **notice that a stated requirement
was not being tested by the stated gate** — which is exactly the failure that
survives review, because everyone reads "RTF < 0.5" as "it's fast" and moves on.

I gave it the target (~7 s word age, from four options it costed out) and it
back-solved the config rather than picking round numbers:

```
D ≤ (7.0 − 0.6 − 0.3) / (1 + 0.5) = 4.07 s   →   max_utterance_ms = 4000
```

That is the difference between a derived value and a plausible one, and the docs
now say which it is.

**Where I pushed back:** it initially wanted to describe this as superseding D12.
It isn't — D12's reasoning about why a cap must exist at all is still correct and
still load-bearing. What changed is that the cap acquired a derivation. The
DECISIONS entry says that explicitly, and D12 is left unedited, which is the
practice D16 established when the platform pivot could have quietly rewritten D6.

## Four decisions I was asked to make

The model costed each option out rather than asking me to pick blind. That is
the shape of question I want from it — the tradeoff made concrete, with a
recommendation, not a menu.

| Decision | Chosen | The cost I accepted |
|---|---|---|
| Latency target | ~7 s word age | more mid-sentence cuts than 8 s utterances |
| Overload behaviour | adaptive shrink → drop oldest | dropped audio is gone, but a visible gap beats silent unbounded lag |
| Caption unit | sentences with carry-over | slightly more moving parts than 1 caption per utterance |
| Caption card | translation primary, source secondary | more UI, but failed translations degrade instead of blanking |

## What it caught that I would not have

- **Sentence carry-over is free.** D12 had already accepted that max-length cuts
  damage translation quality. The fix — publish complete sentences immediately,
  carry only the trailing fragment into the next utterance — costs nothing on the
  latency budget because only the fragment waits. It then immediately named the
  bug in its own proposal: a held fragment at the end of a meeting is never
  emitted, so the flush rule went into the design before the code existed.
- **Windows uses `spawn`, not `fork`.** The worker re-imports the package and
  loads the model cold, so "Start" looks like a hang without a ready handshake.
  This is exactly the class of thing that would have cost an evening on the
  Windows machine, found by reasoning about the platform rather than by hitting it.
- **Out-of-order captions.** Transcripts leave the single worker in FIFO order,
  but concurrent translations do not finish in order. One serialized translation
  task fixes it for free.
- **Translation belongs with the server, not the GPU worker.** D15 said
  "transcription in a separate process" and I had read that as "the heavy stuff
  goes in the worker". Translation is network-bound; behind the GPU it would idle
  the scarcest resource waiting on the least scarce one.

## What I learned

Session 1's lesson was measure, don't assume. Session 2's was that when the thing
you measured changes, the conclusions drawn from it do not update themselves.
This session's is narrower and sharper: **a metric you are measuring honestly can
still be the wrong metric.** RTF was never wrong — it was correctly defined,
correctly gated, and genuinely load-bearing for queue stability. It just did not
answer the question the Definition of Done was asking, and having a rigorous
number made it *less* likely anyone would notice, not more.

The practical version: for every requirement in the brief, name the metric that
would fail if the requirement were violated. "Minimal latency" had no such
metric. It has one now, and it is in the DoD as a number.

## Next

Build `AudioSource` (WASAPI loopback + WAV) and the `Segmenter` at the derived
values, then run the benchmark on Windows against three gates instead of two.

---

**Screenshots to capture next session:** `list_devices` showing the WASAPI
loopback device, a captured 16 kHz mono WAV in a waveform viewer, and
`dump_utterances` output showing utterance boundaries at
`max_utterance_ms = 4000` against live meeting audio.
