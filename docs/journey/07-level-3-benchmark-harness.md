# Journey 07 — Level 3's harness, and the number that does not exist yet (2026-08-19)

Session with Claude (Opus 5) in Claude Code. Fourth session that wrote code, and
the first whose deliverable is an instrument rather than a result.

The whole session is shaped by one asymmetry: Level 3 is a **measurement**, the
measurement needs a GPU and a ten-minute recording, and this laptop has neither.
So the session built the half that can be built here and was explicit — in the
code, in the docs and below — about the half that cannot.

## The prompt that started it

Same shape as sessions 5–7. Several lines arrived truncated in transmission;
they are marked `[…]` rather than tidied up, because a journey doc that quietly
repairs its own prompts is not evidence of anything.

> New session on the speech translator. Level 2 (Segmentation) landed and is
> pushed on branch `build` (commit 649740a). We are building Level 3's harness —
> `tools/benchmark.py` — and only that. CLAUDE.md loads automatically; read these
> in order before doing anything:
>
> 1. docs/STATE.md — where we are. Note that Level 0's and Level 1's Windows
>    halves are STILL owed, and that Level 1's 10-minute record_loopback capture
>    is this level's input, so Level 3 cannot produce a single number until it
>    exists.
> 2. docs/PLAN.md — Level 3's deliverables and acceptance criteria.
> 3. docs/BENCHMARK.md — this is the spec. The harness exists to execute this
>    document: 7 configurations, 10 sustained minutes each, per-chunk timing,
>    first-minute vs last-minute RTF, VRAM polled across the run, and the results
>    table at the end.
> 4. docs/DECISIONS.md — D18, D19, D20, D25, D26, D36, D41, D43, D51, D52 bind
>    this level. D53-D56 are […] rest; do not re-litigate them.
>
> Task: build `speech_translator/tools/benchmark.py`, per D51 — written and
> dry-run on Linux, executed on Windows, with the […] about it being that a GPU
> answers. Nothing else in Level 3 is code: the model choice is a measurement and
> it is not yours to m[…]
>
> Scope, explicitly: the harness ONLY. Do not buil[d …] a worker process, do not
> write `model_size` or `compute_type` into config.py. Those are Level 4 and the
> benchmark's output res[…] takes a configuration as arguments and reports; it
> decides nothing.
>
> Constraints I don't want re-derived:
> • The run configuration is fixed by BENCHMARK.md […] config.py: beam_size=1,
>   condition_on_previous_text=False, vad_filter=False, language pinned.
>   Benchmarking anything else pre[…] we are not shipping.
> • Model load time is excluded from RTF and repor[…]nk is warm-up and reported
>   separately (step 5). RTF is over the whole run, not a slice (step 6).
> • First-minute vs last-minute RTF is not a nice-to-have — the difference is the
>   throttling penalty and it is the number that p[…]
> • `MT_roundtrip` in the gate-3 formula must be MEASURED, not the guessed 300 ms:
>   ten real calls, take the p95 (D51). `doctor` a[…] the code exists. If there is
>   no API key, record that the term is missing — do not silently substitute a
>   constant into a g[…]
> • Gate 3's column is labelled **predicted** in every output the harness emits.
>   A prediction reported as a measurement is the […]
> • On Windows, `windows_cuda.add_cuda_dll_directories()` must run before
>   faster-whisper is imported (D41, D43), and the […] anything until
>   `ctranslate2.get_cuda_device_count()` returns 1 (D36 — there is no CPU
>   fallback, so a silent fallback would pre[…]).
> • 70 minutes of runs. A configuration that OOMs must be recorded as a failed row
>   and the harness must continue to the next one […] because configuration 3 died
>   is the failure D51 wrote this file to prevent. Results are written
>   incrementally for the same […]
> • Emit the results as JSON so the BENCHMARK.md table is transcribed, not retyped.
> • Force UTF-8 (D41/D42) and use the existing `op[…]seam for the audio — do not
>   open a WAV by hand.
>
> Two things I expect to be genuine decisions rather than defaults, and I want
> them argued in DECISIONS.md rather than chosen quietl[y]:
> • BENCHMARK.md step 3 says chunk at a fixed 4 s, with a clause beginning "if
>   Level 2 is not built yet". Level 2 IS built no[w …] D52's stated cost — an
>   optimistic RTF from an unrealistically long chunk distribution — no longer has
>   to be paid. Decid[e …] chunks with, say why, and reconcile BENCHMARK.md and
>   D52 to match.
> • How the Linux dry run works without a GPU and […] model on CPU needs a weights
>   download; a stub needs no network but times nothing real. Pick one, or both
>   behind a flag, […] parts of the harness a dry run actually exercises and which
>   stay untested until Windows.
>
> I develop on Linux and test on Windows via git, […] and D51 applies. Be precise
> about what that means here: everything except "a GPU answers" can be proven on
> this box — the chunkin[g …] arithmetic, the first/last-minute split, the p95,
> the incremental writes, the OOM path, the table rendering. Test those. What[…]
> measurement itself, and there is a second reason it cannot: **the input WAV does
> not exist yet**, because record_loopback has not[…] not fabricate a results
> table, do not fill in BENCHMARK.md's hardware survey, and do not tick Level 3's
> acceptance boxes.
>
> Run the tests as `PYTHONPATH= uv run pytest` on […] see STATE.md's "Known
> environment quirks", deliberately not worked around in project config).
>
> Finish by telling me which of Level 3's acceptan[ce criteria are] reachable
> versus which are owed to the Windows machine and why; then update PLAN.md's
> status column, docs/STATE.md and docs/s[…] changes the method, append to
> docs/DECISIONS.md for the two decisions above, and write docs/journey/07-*.md
> with the prompts and […] doc is a graded deliverable. Then commit to `build`.
>
> Still owed and unchanged: Level 0's `uv sync` + `doctor` on Windows, Level 1's
> 10-minute record_loopback with its playback chec[k …] boundary inspection
> (`dump_utterances --write-wav` over that same recording). The recording now
> blocks three things at once, i[…] three visible in STATE.md.
>
> Still blocked, unchanged: the truncated scope line in the assignment brief
> (PRD.md) and the deadline. Don't design around them.

## Two questions asked before any code

The prompt named two decisions it wanted argued. Reading `BENCHMARK.md` against
D51 turned up two more forks that the documents genuinely did not settle, and
both change the numbers rather than the style, so they were put to the user
before planning finished.

**1. What does "ten sustained minutes" mean?** `BENCHMARK.md` says "run for at
least 10 minutes continuously per configuration". D51 and `PLAN.md` both price
the matrix at ~70 minutes. Those only reconcile one way: ten minutes of *audio*
at RTF 0.3 is a three-minute run and the whole matrix is ~25 minutes — and a GPU
that never gets hot cannot report a throttling penalty, which is the number
`BENCHMARK.md` says predicts demo behaviour. Answer: **wall clock, loop the
audio**. That is D59.

**2. What is gate 3's p95 a percentile *of*?** With Segmenter chunking there is a
real distribution of utterance duration `D` and of per-chunk RTF, so "p95 word
age" can mean the 95th percentile over the chunks, or a point estimate at the
4 s cap. Answer: **p95 over the real chunks**, with the cap bound reported beside
it — the first is a genuine percentile over what the app will produce, the second
is the arithmetic D25 used to derive `max_utterance_ms` in the first place, and
neither is a substitute for the other.

Both answers came back as the recommended option. Worth recording that they were
*asked*: the first one in particular would have been very easy to implement the
cheap way, produce a table, and never notice that the throttling column was
measuring nothing.

## The three decisions

### D57 — chunk at the Segmenter's real distribution

`BENCHMARK.md` step 3's escape clause ("if Level 2 is not built yet") no longer
applies, so the harness drives the real `Segmenter` over the recording. Three
reasons, and the third was the one that changed the design:

1. A fixed 4 s slice is not the workload. It contains the closing silence the
   Segmenter trims and lacks the pre-roll it prepends (D55) — same file,
   different bytes, different cost.
2. It removes the optimism D52 accepted on the way in, and records the
   distribution rather than assuming it.
3. **The chunks are cut once and shared.** The Segmenter is deterministic over a
   WAV, so all seven configurations get byte-identical audio. Otherwise an RTF
   difference between two rows could be a difference in what was measured. This
   also puts the VAD's own cost outside the timed region.

D52's bracket is kept for the selected row rather than dropped, and D52 itself is
not edited — the project supersedes by appending (D12 under D25, D6 under D16).

### D58 — both dry-run engines, and the stub is the default

The stub makes the dry run part of `pytest` rather than a ritual. The real CPU
model is what catches the thing a stub cannot: `model.transcribe()` returns a
**lazy generator**, and a harness that times the call without draining it reports
an RTF near zero while remaining perfectly self-consistent. The generator is
drained inside `FasterWhisperEngine.transcribe`, and there is a test with a fake
`WhisperModel` returning a genuinely lazy generator that asserts it was consumed
— no weights required.

### D59 — wall clock, looped

Above. The stated cost is that back-to-back is a heavier duty cycle than the live
app, whose GPU idles between utterances. That overstates thermal load, which is
the right direction to be wrong in for a gate that exists to catch a machine
flattering itself when cold.

## Two bugs found while writing, both the same shape

Neither would have failed a test on this laptop, and both are the kind that
surface at minute 55 on the machine you have least access to.

**The GPU samples were attached to the row after the row was summarised.** The
teardown lived in a `finally:` block and the summary was computed in the
`return` expression above it — so Python evaluated the summary first, and every
row would have reported no VRAM at all. Fixed by moving the metrics out of the
try/finally entirely and computing them after the poller has stopped. The second
half of the same bug: the `finally` block *rebound* the sample list rather than
extending it, so the reference already returned was the empty one. Both are
noted in comments, because the correct-looking version is the one that comes
naturally.

**`laps` was off by one.** It was computed from the last record's position rather
than from the count, so sixty calls against a four-chunk recording reported 14.75
laps instead of 15. Caught by a test that asserted a round number, which is why
the test asserted a round number.

## What the dry run actually proved

Two real invocations on this box, both against `assets/speech_fixture.wav`:

```
$ ... benchmark --input-wav assets/speech_fixture.wav --dry-run
MT: **unmeasured** — GOOGLE_TRANSLATE_API_KEY unset — the MT term of the word-age
formula is missing, not 300 ms (D51); gate 3 reports *unknown*
chunks: 4 · 10.9 s of 13.9 s · p50 3072 ms / p95 3923 ms
  small/float16: RTF 0.250 (first 0.250 / last 0.250) · peak VRAM — ·
                 word age (predicted) MT unmeasured
```

```
$ ... benchmark --engine faster-whisper --device cpu --configs tiny:int8 \
      --minutes 0.5 --language en --no-mt
  tiny/int8: RTF 0.137 (first 0.137 / last 0.137) · peak VRAM — ·
             word age (predicted) MT unmeasured
```

The second one downloaded real weights, loaded a real `WhisperModel`, and decoded
82 chunks over 30 seconds — `load_s` 27.9 s (mostly the download, and excluded
from RTF), warm-up 0.44 s reported separately, `rtf_p95_chunk` 0.39 against a
whole-run 0.137. The transcripts came back as `"Hello everyone, can you all hear
me okay?"` and `"Yes."`, which is the espeak-ng fixture, correctly. **RTF 0.137
rather than 0.002 is the evidence that the generator was drained.**

Neither of these is a benchmark result and the harness says so itself: every row
of a non-CUDA run is stamped `NOT A GPU MEASUREMENT`, in the JSON and in the
rendered table.

49 new tests, 153 total, no GPU, no weights, no network.

## What this session deliberately did not do

- **No results table.** `BENCHMARK.md`'s hardware survey and results table are
  still empty. A 13.9 s fixture looped twenty times exercises the machinery; it
  is not a measurement of anything.
- **No acceptance boxes ticked.** All five of Level 3's are measurements. The
  fifth ("the harness runs on Linux against a WAV") is the only one this session
  moved at all, and it is recorded as *partially met*, because the WAV it was
  written for does not exist.
- **No model selected, and no seam to select one.** `config.py` still has
  `model_size = None` and `compute_type = None`. There is a test asserting the
  harness contains no way to write them.

## The thing that is now clearly the bottleneck

One 10-minute `record_loopback` capture on the Windows machine unblocks four
separate things: Level 1's playback check, Level 2's boundary inspection, Level
3's entire measurement, and — alongside `uv sync` and `doctor` in the same
sitting — Level 0's Windows half. It has been outstanding since session 5. That
table is now in `STATE.md` rather than spread across four bullets, because four
bullets is how something stays outstanding for four sessions.
