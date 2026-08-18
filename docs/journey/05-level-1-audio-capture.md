# Journey 05 — Level 1, and proving the test can see the bug (2026-08-18)

Session with Claude (Opus 5) in Claude Code. Second session that wrote code, and
the first that wrote code with a failure mode I already knew the name of before
we started.

## The prompt that started it

> New session on the speech translator. Level 0 (Foundation) landed and is
> pushed on branch `build` (commit 6403b6f). We are building Level 1 (Audio
> capture). CLAUDE.md loads automatically — read these in order before doing
> anything:
>
> 1. `docs/STATE.md` — where we are; note the "Known environment quirks" section
> 2. `docs/PLAN.md` — Level 1's deliverables and acceptance criteria
> 3. `docs/INTERFACES.md` — §1 AudioSource is the contract you are
>    implementing. You should not need to read another stage.
> 4. `docs/DECISIONS.md` — D3, D17, D22, D23, D24, D36, D41 bind Level 1.
>    D42-D45 are what Level 0 added. Skim the rest; do not re-litigate them.
>
> Task: build Level 1 (Audio capture), per `docs/PLAN.md`. Deliverables are the
> `AudioSource` Protocol, the format layer (decode → downmix → streaming
> resample → exact 512-sample frames, all *inside* the source per D24),
> `WavFileSource` (realtime and as-fast-as-possible), `WasapiLoopbackSource`,
> and the `tools/list_devices` + `tools/record_loopback` commands.
>
> Constraints I don't want re-derived:
> • 16 kHz mono int16, 512-sample / 1024-byte frames — the constants already
>   exist in `speech_translator/config.py`; use them, don't redefine them.
> • Resampling uses `soxr.ResampleStream` — ONE instance for the life of the
>   stream, so filter state carries across chunks. Rebuilding it per chunk puts
>   clicks at every boundary and that is the failure this level is most likely
>   to ship with.
> • Never hard-code 48000. Read the device's `defaultSampleRate` (D41).
> • `pyaudiowpatch` imports lazily, inside the function — the package must still
>   import cleanly on Linux.
> • The 16 kHz mono path must be bit-exact — no float round-trip when no
>   conversion is needed.
> • No PipeWire/Linux capture source (D17). `WavFileSource` is the Linux path.
> • No torch, ever (D35).
>
> I develop on Linux and test on Windows via git, so the same split as Level 0
> applies: unit tests and the WAV path are verifiable here; the device and the
> 10-minute recording are Windows-only and get carried forward honestly rather
> than ticked off from here.
>
> Note: `record_loopback` is on the critical path for `BENCHMARK.md` — it
> sources its sample audio through it, so it needs to produce a file the Level 3
> measurement can actually use.
>
> Run the tests as `PYTHONPATH= uv run pytest` (this box has a global
> PYTHONPATH — see STATE.md). Finish by running Level 1's acceptance criteria
> and telling me which pass, which are owed to the Windows machine, and why.

Same shape as the Level 0 prompt, and for the same reason: five sessions of
decisions exist so that they can be *spent*, not re-argued.

## What got built

`speech_translator/audio/` — `source.py` (the Protocol and four error types),
`format.py` (`FrameFormatter`), `wav_source.py`, `wasapi.py`, and `select.py`,
which is the single place in the codebase allowed to ask what platform it is on.
`speech_translator/tools/` — `list_devices` and `record_loopback`. 34 new tests;
50 pass on this laptop with no GPU, no audio device and no network.

```
$ PYTHONPATH= uv run python -m speech_translator.tools.record_loopback \
      --input-wav in48.wav --out out16.wav
recording -> out16.wav
source: in48.wav · 10.00 s · 48000 Hz 2 ch int16 → 16000 Hz mono int16 · streaming resample · 512-sample frames
   10.0 / 600 s    313 frames  [##############......] peak  -6.3 dBFS
wrote out16.wav
  10.02 s · 313 frames × 1024 bytes · 16000 Hz mono int16
  peak: -6.3 dBFS
```

```
$ PYTHONPATH= uv run python -m speech_translator.tools.list_devices ; echo $?
pyaudiowpatch is not available. It is a Windows-only dependency (marked
sys_platform == 'win32' in pyproject.toml) and WASAPI loopback exists only on
Windows (D3, D17). On Linux use WavFileSource — that is the supported dev path (D22).
2
```

That second output is the whole D22 discipline in six lines: the command exists,
it runs, it refuses honestly, and it names the alternative.

## The thing I actually wanted from this session

I told it in the prompt that a per-chunk resampler was the most likely way this
level ships broken. What I did not expect was to get that claim turned into a
number before a line of `FrameFormatter` was written.

It measured first — 3 s of a 440 Hz tone at 48 kHz, pushed through in random
1–5000-sample chunks, one streaming resampler versus a fresh one per chunk:

| | output samples (expect 48 000) | max sample-to-sample jump |
|---|---|---|
| One `ResampleStream` | 48 000 | 3 453 |
| A fresh one per chunk | 48 003 | **9 087** |
| Steepest slope this tone can have | — | 3 505 |

And later, on a 10 s two-tone signal through the finished tool, the same
comparison in the frequency domain: **−81.4 dB** of out-of-band energy streaming,
**−30.2 dB** per chunk. That 51 dB is what "clicks at chunk boundaries" is,
written as a number instead of an adjective.

Then it did the part I want to keep. The test suite contains
`test_control_a_per_chunk_resampler_does_produce_seams`, which *deliberately
reproduces the bug* and asserts that the seam bound rejects it. Without it, a
threshold loose enough to accept anything would have passed silently and I would
have had a green suite that proved nothing. And to check the guard was wired to
the real code rather than to the test's own reimplementation, it injected the
per-chunk rebuild into `FrameFormatter` and re-ran:

```
=== with the bug injected ===
FAILED tests/test_audio.py::test_streaming_resampler_has_no_chunk_seams
FAILED tests/test_audio.py::test_resampler_is_constructed_exactly_once
2 failed, 32 passed
=== restored ===
34 passed
```

Two tests fail, and only those two. That is a much stronger claim than "the
tests pass", and it cost about two minutes.

## Two scope questions it asked before starting

Both were genuinely mine to answer, and I would rather be asked than have them
assumed:

1. **Should `record_loopback` also write a device-native copy?** Its argument
   for: if the Windows recording sounds wrong, a raw capture is what separates a
   device problem from a resampler problem. I said no. The cost is real and is
   written into D47 rather than left implicit — but the resampler side is now
   the best-measured thing in the repo, so the diagnosis is not actually
   ambiguous.
2. **Blocking `stream.read`, or a callback with a bounded queue?** I took the
   blocking read. D28 already owns backpressure downstream, and a second
   drop policy at the capture end would mean two things that can lose audio for
   unrelated reasons. D50 records the condition for revisiting — overruns in the
   ten-minute Windows run — and the fact that the swap is local to one file.

It also proposed `--input-wav` unprompted, and that one earned its place: it
means the metering, the incremental WAV writing, the Ctrl-C path and the
duration report all *run* on this laptop. What is left untested here is opening
a device, which is untestable here by definition. That is D48, and it is the
difference between "Windows-only" as a fact and "Windows-only" as an excuse.

## What did not go smoothly

**Two tests passed for the wrong reason and I made it say so.** The first draft
of `test_wav_source_close_releases_the_file` asserted
`src._wav._file is None or getattr(src._wav, "_file", None) is None` — which is
a tautology dressed as a check; it cannot fail. The float-saturation test used a
strided sample of the output compared to a tolerance, which is a statistical
argument where an exact one was available. Both are now element-wise: the close
test holds the real OS file object and asserts `handle.closed`, and the
saturation test compares the whole output array to an exact expected array,
because at 16 kHz in and 16 kHz out there is no resampling and samples map one
to one. A green test that cannot go red is worse than no test, because it
occupies the space where a real one would have gone.

**`wave` cannot read float32 WAVs, and we found that out by trying.** Python's
stdlib reader raises `Error: unknown format: 3`. That matters because WASAPI may
hand us float32 and I had half-assumed the file path and the device path were
symmetric. They are not. `WavFileSource` now raises `UnsupportedAudioFormat` with
a message naming the format and pointing at `record_loopback`, and `av` — already
in the tree as a `faster-whisper` dependency — is noted as the escape hatch if a
float WAV ever actually turns up. Not built, because it is not needed yet.

**Running the tool for real found a traceback that the tests did not.** A
mistyped `--input-wav` path produced a raw `FileNotFoundError` stack rather than
a message. One line to fix, and the only reason it surfaced is that the tool was
actually executed against a wrong path instead of only being unit-tested against
right ones.

## My assessment of the output

**What I rate highly.** The 16 kHz mono path is bit-exact *by construction*
rather than by tolerance — when the input is already the target format, no
resampler is built, no downmix runs and no float conversion happens, so the
bytes are the same bytes and the test can assert byte equality. I had asked for
"no needless float round-trip"; what I got was an implementation where the
round-trip is structurally impossible rather than avoided by an `if`.

Second, `record_loopback` writes the *normalised* stream rather than the device
stream (D47), and the reasoning it gave is better than the one I had: not only
does `BENCHMARK.md` want 16 kHz mono, but writing the device format would have
made the recording a recording *of the device*, proving nothing about our code.
Writing the pipeline's own bytes makes the file itself the evidence for the
format layer.

**Where it fell short.** The first draft of `record_loopback` leaked the audio
source if the output path was unwritable — it opened the source, then opened the
WAV, with no handler between them. Found by re-reading, not by a test, and there
is still no test for it. Minor, and I am recording it rather than quietly fixing
it, because "found by re-reading" is not a process I can rely on twice.

**Where I would push back.** `FrameFormatter` handles uint8, int16, 24-bit,
int32, float32 and float64 inputs, and the demo machine will produce exactly one
of those. That is speculative generality, and on a different project I would cut
it. Here it is about fifteen lines inside one already-necessary conversion step,
and the alternative — discovering on the demo machine that the endpoint reports
something we did not handle — costs a git round trip to find out. I am letting
it stand as insurance, not as a principle.

## Decisions taken

| | Chosen | The reason that survives |
|---|---|---|
| D46 | One `ResampleStream` per source; 16 kHz mono bypasses it entirely | −81.4 dB vs −30.2 dB out-of-band, and a control test proving the guard can see the bug |
| D47 | `record_loopback` writes the normalised stream, not the device stream | `BENCHMARK.md` consumes it, and it makes the recording the evidence for the format layer |
| D48 | `--input-wav` as the Linux smoke path | leaves *opening a device* as the only genuinely Windows-only part of the tool |
| D49 | Ask for `paInt16`, accept `paFloat32`; read `defaultSampleRate` | the demo machine decides both, and 48000 appears nowhere in the audio package |
| D50 | Blocking `stream.read` over callback + queue | D28 already owns backpressure; the revisit condition is written down, not implied |

## Level 1 acceptance — what actually passes

| Criterion | Result |
|---|---|
| Every frame exactly 1024 bytes; sample count within one frame of duration | **pass** — 16 k/44.1 k/48 k/96 k, mono and stereo, ragged chunks |
| 48 kHz stereo and 16 kHz mono give identical-shaped output; 16 kHz path bit-exact | **pass** — byte equality, not tolerance (D46) |
| `record_loopback` captures 10 min on Windows that plays back cleanly | **owed.** Pitch and noise floor verified offline through the real tool — peaks at exactly 1000/2500 Hz, out-of-band 81.4 dB down. The *device* is what is untested |
| Unit tests pass on Linux with no audio hardware | **pass** — 34 new, 50 total |

Three of four; the fourth needs the demo machine, and it is the same recording
Level 3 needs as its benchmark input, so it is wanted twice over.

Still owed from Level 0 and unchanged: `uv sync` + `doctor` on Windows. It has
now been outstanding for two sessions, which is exactly how a carried item turns
into a forgotten one, so it is at the top of `STATE.md` → Next with the Level 1
commands underneath it.

## What I learned

**A test that cannot fail is worse than no test**, and the only way to know
which kind you have is to break the code and watch. The mutation run took two
minutes and converted "the seam tests pass" into "the seam tests fail when, and
only when, the seam exists". Every other test in this file I now trust slightly
less, because none of them have been through that.

**Measuring the failure before writing the fix changed what got built.** The
−81.4 / −30.2 dB pair is not decoration — it set the threshold the test uses.
Had we written the formatter first and the test after, the threshold would have
been chosen to make the existing implementation pass, which is the same shape of
mistake as the tautological close test, one level up.

**"Windows-only" is a claim that should keep shrinking.** At the start of the
session the honest statement was "the capture tool is untested until I get to
the other machine". After `--input-wav` it is "opening the endpoint is untested".
Same amount of Windows-specific code, much smaller untested surface, and the gap
between those two sentences is where D22 either works as a discipline or quietly
becomes an excuse.

## Next

Level 2 — segmentation. But first, on the Windows machine, and now with two
levels' worth of debt on it: `git pull`, `uv sync`, `doctor`, `list_devices`,
then a ten-minute `record_loopback` of a Spanish podcast. Three unknowns to write
down while there — whether `uv sync` agrees with the lockfile, what
`defaultSampleRate` the endpoint reports, and whether it gives us `paInt16` or
falls back to `paFloat32` (D49).

---

**Screenshots to capture next session:** `list_devices` on Windows next to the
Linux refusal above — that pair says more about the dev/test split than a
paragraph does; and the waveform of the 10-minute capture, because "no clicks" is
a claim best made with a picture.
