# Journey 06 — Level 2, and the component that was dead the whole time (2026-08-18)

Session with Claude (Opus 5) in Claude Code. Third session that wrote code, and
the first where the interesting output was not the code.

## The prompt that started it

Same shape as sessions 5 and 6. Several lines arrived truncated in transmission —
marked `[…]` below rather than tidied up, because a journey doc that quietly
repairs its own prompts is not evidence of anything.

> New session on the speech translator. Level 1 (Audio capture) landed and is
> pushed on branch `build` (commit 98390ad). We are building Level 2
> (Segmentation). CLAUDE.md loads automatically — read these in order before
> doing anything:
>
> 1. `docs/STATE.md` — where we are; note "Known environment quirks" and that
>    both Level 0's and Level 1's Windows halves are still owed
> 2. `docs/PLAN.md` — Level 2's deliverables and acceptance criteria
> 3. `docs/INTERFACES.md` — §2 Segmenter is the contract you are implementing,
>    and §1 is the input you are consuming. You should not need to read any
>    other stage.
> 4. `docs/DECISIONS.md` — D11, D12, D23, D25, D28, D30, D35 bind Level 2.
>    D46-D50 are what Level 1 added. Skim the rest; do not re-litigate them.
>
> Task: build Level 2 (Segmentation), per docs/PLA[…]ilero wrapper over the
> vendored ONNX, the `Segmenter` (close on silence OR max-length, with a runtime
> override of the effective max), t[…]ctly as INTERFACES.md §2 defines it, the
> `speaking` side channel, and tools/dump_utterances.
>
> Constraints I don't want re-derived:
> • Silero runs from assets/silero_vad.onnx via onnxruntime. Never `silero-vad`,
>   never torch (D35). The signature is in INTERFA[…]ile is already verified by
>   `doctor` and by tests/test_foundation.py.
> • VAD state carries between frames — [2, N, 128][…]very call. This is the same
>   class of bug as Level 1's resampler: state that must survive a boundary.
>   Treat it that way, and tes[…]
> • Frames arrive from AudioSource already normalised — 16 kHz mono int16,
>   exactly 512 samples (D24, D46). Do not resample, do no[…] sample-rate branch.
>   Consume `frames()` and nothing else.
> • The D25 values already exist in speech_transla[…]n't redefine them.
> • The Segmenter must accept a *runtime* override of effective max_utterance_ms,
>   because Level 6's backpressure controller shr[…]ild the seam now even though
>   nothing calls it yet.
> • min_utterance_ms discard is a drop, not an emi[…]is populated on every
>   utterance. `speaking` is a side channel, not a field on Utterance (D11).
>
> Two things I expect to be genuine decisions rath[…] quietly, and I want them
> labelled the way config.py labels the D30 thresholds — TUNABLE, not derived:
> • The speech-probability threshold. Silero emits a probability; nothing in D23
>   or D25 fixes where the cut is.
> • Whatever you do about flicker at that boundary — hysteresis, a minimum speech
>   run, or a deliberate decision not to. Say whic[…]
>
> I develop on Linux and test on Windows via git, [D2]0-1 applies. Note what that
> means here specifically: PLAN.md's first acceptance criterion is "on a captured
> meeting WAV, boundar[…] inspection", and that WAV does not exist yet —
> record_loopback has not run on the demo machine. Do not fake it and do not
> quietly [drop] for it. Build what can be proven here (silence-only and
> music-only produce zero utterances, no utterance exceeds the max, sub-30[0 ms
> blips discarded, VAD] cost near 0.156 ms/frame), and carry the real-speech
> inspection forward honestly.

The truncation cost nothing, which is itself a data point: five sessions of
`DECISIONS.md` meant the missing halves of those sentences were recoverable from
the repo. A prompt that leans on committed decisions degrades gracefully.

## The find

**The VAD did not work, and had not worked since Level 0.**

`INTERFACES.md` §2, D35, `assets/README.md` and `doctor` all documented the
model's input as `[N, 512]`. The vendored file is Silero **v5**, whose 16 kHz
call is **576** samples wide: the 512 new ones with **64 samples of the previous
frame prepended**.

The ONNX input dimension is dynamic. So the wrong call runs, returns a float,
and raises nothing:

| call shape | silence | 440 Hz tone | music chord | white noise | **real speech** |
|---|---|---|---|---|---|
| 512, as documented | 0.001 | 0.001 | 0.001 | 0.002 | **0.0005** |
| 576 = 64 context + 512 new | 0.004 | 0.000 | 0.006 | 0.011 | **1.000** |

The right-hand column is `/usr/share/sounds/alsa/Front_Center.wav` — a real
recorded human voice that ships with ALSA, found while looking for anything on
this laptop that Silero ought to fire on.

**Why it survived three sessions.** Every check that existed was a shape check.
`doctor` constructed the session, pushed one frame through, and reported the
timing — a check that the *file loads*, dressed as a check that the *VAD works*.
`tests/test_foundation.py` asserted the output tensor's shape. Both passed. Both
would have passed against a model that returns a constant.

**How it would have failed.** A VAD that answers "no speech, ever" emits no
utterances, so no transcripts, so no captions — with every stage reporting
healthy. The symptom on the demo machine is a blank screen and a green badge,
and the natural suspects are all upstream (capture) or downstream (CUDA, model
download) of the actual fault. Best case that costs an evening; worst case it
gets found during the demo.

This is D46's failure one layer up — **state that must survive a boundary,
discarded at the boundary** — and the prompt named that class of bug before the
session started. It just named one piece of state, and there were two.

## What got built

`speech_translator/segment/` — `vad.py` (the `SpeechDetector` Protocol and
`SileroVad`, which owns the ONNX session, the LSTM state and the 64-sample
context) and `segmenter.py` (`Utterance` and `Segmenter`). Plus
`tools/dump_utterances`, `tools/make_speech_fixture`, and
`assets/speech_fixture.wav`. 54 new tests; **104 pass** on this laptop with no
GPU, no audio device and no network.

```
$ PYTHONPATH= uv run python -m speech_translator.tools.dump_utterances \
      --input-wav assets/speech_fixture.wav --write-wav utts/
source:    speech_fixture.wav · 13.92 s · 16000 Hz mono int16 → passthrough (bit-exact, no resampler)
vad:       silero v5 · silero_vad.onnx · 576-sample calls (64 context + 512 new) · 16000 Hz · CPU, 1 thread
thresholds: open 0.5 / release 0.35 · silence 600 ms · max 4000 ms · min 300 ms
id      start_ms   end_ms  dur_ms  closed_by   gap_ms     peak
u_0001       416     3072    2656  silence        416     -0.8
u_0002      3840     4576     736  silence        768     -3.2
u_0003      5408     9408    4000  max_length     832     -2.3
u_0004      9408    12896    3488  silence          0     -2.2
4 utterances · 0 discarded (< min_utterance_ms, a VAD blip not speech — D30)
  435 frames · 13.92 s audio · 9.18 s speech (66%)
  VAD cost: 0.141 ms per 32 ms frame (D35 measured 0.156 — this is the number Level 2's acceptance asks for)
```

The fixture's silences are 900, 900 and 1000 ms **by construction**, so those
`gap_ms` values are a real check and not a shrug: the boundaries landed in the
pauses that were put there. `u_0003` is the max-length cut, and `u_0004` carries
`gap_ms = 0` because a max-length close reopens contiguously — Level 5's
fragment carry-over depends on that audio being unbroken.

## The speech problem, and why there is now a WAV in the repo

Silero rates silence, a 440 Hz tone, a four-note chord and white noise below
0.05. Every *negative* test is therefore self-contained, and the two acceptance
criteria about silence and music were easy.

**No positive test was possible.** Nothing numpy can synthesise reliably crosses
0.5 — formant-synthesised vowels with a syllable-rate envelope were tried and
peak around 0.8 on 6% of frames, which is not something to assert on. And
without a positive test, the stage can be dead and green. Which it was.

Four options, and the one chosen was to **generate speech and commit it**:
`libespeak-ng` driven through ctypes, three phrases separated by known 900–1000 ms
silences, 13.92 s at 16 kHz mono int16, byte-reproducible across runs. Formant
synthesis, so the output embeds no recorded audio and no third-party sample and
there is no licence to reason about. The alternative — generate at test time if
espeak-ng is present, else skip — skips on Windows, which is the machine that
matters.

Two things this immediately caught that would otherwise have shipped:

1. `.gitignore` had `*.wav`. The fixture would have been silently absent from
   the commit, and every positive test would have failed on Windows only.
2. `doctor` can now fail on a dead VAD, because it has something to assert
   against: speech > 0.9 and silence < 0.1.

**Its limit, stated where it cannot be missed** (`assets/README.md`, D56,
`PLAN.md`, `STATE.md`): TTS speech has no room tone, no music bed, no
overlapping speakers and no reverb — precisely what D23 chose Silero to survive.
It proves the VAD is alive and the state machine behaves. It is **not** the
captured meeting WAV `PLAN.md` asks for, and that criterion stays open.

## The two decisions I was asked to make out loud

**Threshold: 0.50 to open, 0.35 to stay open** (D54). Hysteresis, not a single
cut, and the reason is measured rather than borrowed: real falling edges run
`1.00 → 0.98 → 0.74 → 0.11`, and mid-word dips land in the 0.4s (0.46 and 0.41
both observed). A single 0.5 cut re-closes the utterance on those frames, which
fragments an utterance mid-word and hands the translator half a clause — the D12
tradeoff, paid for nothing. The 0.15 gap is the one Silero's own reference
iterator uses. Labelled **TUNABLE, not derived**, in the block above D30's.

**Flicker: hysteresis for the segmenter, a 160 ms off-debounce for the
indicator, and deliberately no minimum speech run** (D54). The two things that
flicker are different things and get different mechanisms. A min-run is the
obvious third option and it is the wrong one here: it delays the utterance's
start, which spends latency budget (D25) on a problem D30's discard already
solves for free at the other end.

The 160 ms is a correction, not a preference. At 96 ms the indicator blinked
dark for a single frame mid-phrase, because the natural dip around a plosive
runs to three sub-threshold frames and that is exactly where a 96 ms window
expires. At 160 ms it holds through those and still goes dark at real pauses —
there is still a 96 ms dark gap at the comma in the first phrase, which is
correct, because there *is* a pause there.

## The bug the tests caught before the demo machine could

Pre-roll (128 ms before the trigger, so Whisper is not handed a clipped first
phoneme) plus tail pad (192 ms after the last speech frame) is **320 ms of
padding** — and `min_utterance_ms` is **300**.

So a 96 ms cough produces a 416 ms "utterance", and if the D30 discard tests
`end_ms - start_ms`, *every blip clears the bar* and the guard becomes dead code.
The discard now measures **speech**, between the first and last speech frames.

This was found by writing the blip test and getting an utterance back. It is the
best argument in this session for writing the acceptance criteria as tests rather
than checking them by hand at the end: by hand, "sub-300 ms blips are discarded"
would have been ticked off after watching one obviously-silent file produce
nothing.

A smaller one from the same source: the max-length cap now floors to a whole
frame, because `2000 // 32` is not exact and the first version emitted 2016 ms at
the backpressure floor — breaking "no utterance exceeds `max_utterance_ms`" by
one frame at exactly the moment the system is already under load.

## What did not go smoothly

**The first fixture came out 9.1 s instead of 13.9 s**, with the long third
phrase cut in half, because `espeak_Initialize` was being called per phrase.
Caught only because the same script had been measured at 13.9 s during planning
— i.e. by having a number to compare against, not by looking at the code.

**The `speaking` indicator lit up during the opening silence**, because
`_last_speech_frame` was initialised to `-1` and frame 0 therefore sat inside the
debounce window. Visible only because the first end-to-end run printed the
transition edges rather than just the utterances.

**Two test bounds were wrong on the first run.** The control for the dead VAD
asserted `< 0.01` and the real figure is 0.052 — the bound is now expressed as a
fraction of the speech threshold, which is the thing that actually matters. And
a gap assertion tried to predict the exact millisecond offset from first
principles; it now asserts the boundary lands *inside* the inserted pause, which
is the claim worth making.

## Decisions taken

| | Chosen | The reason that survives |
|---|---|---|
| D53 | Silero v5 needs 64 samples of context; `doctor` must assert an answer, not a shape | 0.0005 vs 1.000 on the same recorded speech, and a preflight that could not fail on a dead component |
| D54 | 0.50 / 0.35 hysteresis, 160 ms indicator debounce, no min speech run | measured falling edges and dips; a min-run would spend latency budget on a problem D30 already solves |
| D55 | Pre-roll 128 ms, tail pad 192 ms, and the min-utterance check measures speech | the padding is 320 ms and `min_utterance_ms` is 300 — testing the padded duration cancels D30 |
| D56 | Scripted VAD for the state machine, generated speech for the wrapper | nothing synthesisable crosses the threshold, so without a committed fixture the stage can be dead and green |

## Level 2 acceptance — what actually passes

| Criterion | Result |
|---|---|
| On a captured meeting WAV, boundaries land at real pauses on inspection | **owed.** That WAV does not exist — `record_loopback` has not run on the demo machine. On the generated fixture, whose pauses are known by construction, boundaries land inside them and each utterance is written out for listening |
| No utterance exceeds `max_utterance_ms` | **pass** — every utterance in every test, including 22 s of unbroken speech; holds at the 2000 ms floor too (1984, not 2016) |
| Silence-only and music-only produce zero utterances | **pass** — silence, tone, chord and white noise, 6 s each through the real Silero |
| Sub-300 ms blips discarded, not emitted | **pass** — and measured on speech, not on the padded duration |
| VAD cost near 0.156 ms/frame | **pass** — 0.116–0.141 ms at the corrected 576-sample width |
| *(not on the list)* the VAD detects speech at all | **pass now; was failing before this session** |

Four of five, plus one that was not asked for and mattered more than the four.

Still owed and unchanged, now for the third session running: Level 0's `uv sync`
+ `doctor` on Windows, and Level 1's 10-minute `record_loopback`. The second
blocks Level 3 and now also closes Level 2's open box, so one recording ticks two
criteria and unblocks a gate.

## What I learned

**A check that cannot fail is worse than no check, and "it ran" is that check.**
Level 1's lesson was that a test must be able to see the bug it guards, proved by
injecting the bug. `doctor`'s Silero check had never been through that, and it
was green against a component that returned 0.0005 on speech for three sessions.
The generalisation I actually trust now: *asserting on a shape is asserting on
nothing.* Assert on an answer, in both directions.

**Four places said 512 and all four were copies of one mistake.** The signature
was written down once during the stack session and propagated into
`INTERFACES.md`, D35, `assets/README.md` and `doctor` — which is exactly what a
decisions log is supposed to enable, and exactly why an unverified claim inside
one is expensive. The docs made the error consistent instead of visible.

**"Not testable here" was a conclusion reached too early.** The honest-looking
move was to test the negatives on Linux and carry every positive to Windows.
Twenty minutes of looking turned up real recorded speech already on the disk and
a synthesiser already installed — enough to turn the whole positive path into
something that runs on both machines forever. It is worth spending a little
effort attacking "this can only be verified on the other machine" before
accepting it, because a carried item that is *never* discharged is
indistinguishable from an untested one.

**Padding interacts with thresholds.** Two independently sensible numbers —
320 ms of padding, a 300 ms minimum — silently cancelled a correctness guard when
combined. Neither decision is wrong; the interaction is. That is an argument for
writing the guard as a test with a concrete blip in it, rather than as a
condition that looks obviously right.

## Next

Level 3 — the benchmark, and it is a **gate**. It cannot start until the
10-minute recording exists, so the Windows session is now genuinely blocking
rather than owed. Order on that machine: `git pull`, `uv sync`, `doctor` (which
now says something new — whether the VAD discriminates), `list_devices`,
`record_loopback -t 600`, then `dump_utterances --input-wav` that recording
`--write-wav utts/` and **listen to the cuts**.

---

**Screenshots to capture next session:** `doctor` on Windows with the Silero line
reading `speech 1.00 / silence 0.01` — the check that could not have said that
last session; and the `dump_utterances` table over ten minutes of real meeting
audio next to the fixture table above, because the difference between those two
tables is the whole of what is still owed.
