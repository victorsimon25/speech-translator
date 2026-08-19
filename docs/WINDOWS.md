# Windows runbook — closing Levels 0, 1, 2 and running Level 3

This file exists because four levels are waiting on one machine, and the work is
not hard — it has simply never been done. Everything below is executed on the
**Windows demo machine** (GTX 1650 Ti, 4 GB). The Linux box wrote all of it and
can verify none of it.

Read `docs/STATE.md` first for where the project actually is. This file says what
to *do* on this machine and what to write down afterwards.

---

## The prompt

Paste this into Claude Code on the Windows machine, in the repo root.

```text
New session on the speech translator, running on the WINDOWS demo machine for
the first time. Branch `build`. CLAUDE.md loads automatically.

Read, in this order, before doing anything:
  1. docs/WINDOWS.md   — the runbook for this session. You are executing it.
  2. docs/STATE.md     — where the project is, and what this machine owes it.
  3. docs/PLAN.md      — Levels 0-3 acceptance criteria; these are the boxes
                         this session can finally tick.
  4. docs/BENCHMARK.md — the spec Level 3 executes. The harness
                         (speech_translator/tools/benchmark.py) exists to run
                         this document; do not reimplement any of it.
  5. docs/DECISIONS.md — D18, D19, D20, D25, D26, D36, D41, D43, D51, D52 and
                         D57-D59 bind this session. Do not re-litigate them.

Task: execute docs/WINDOWS.md end to end. In order: verify Level 0 (`uv sync`,
`doctor`), close Level 1 (list_devices, a 10-minute record_loopback capture),
close Level 2 (dump_utterances --write-wav over that capture), then run Level 3
(the benchmark: 7 configurations x 10 wall-clock minutes, ~70 minutes plus
model downloads). Then transcribe the results and update the docs listed in
docs/WINDOWS.md.

Rules I do not want re-derived or worked around:
- Everything is a MEASUREMENT. Report what the machine says, including when it
  is bad news. Never estimate, never fill a cell you did not observe, never tick
  an acceptance box you did not earn. A documented negative result is worth more
  than an assumption that happened to hold.
- The benchmark harness reports; it does not choose. Selecting the model is a
  human judgement — "the fastest that is accurate enough" (D25/D26) — and
  "enough" is decided by LISTENING to the sample transcripts in results.json,
  not by reading the RTF column. Ask me before writing the choice into config.
- A configuration that OOMs is a failed row and the matrix continues. Do not
  restart the whole matrix because row 3 died (D51).
- No CPU fallback exists and none is to be built (D36). If
  ctranslate2.get_cuda_device_count() is not 1, STOP and diagnose — do not pass
  --device cpu to make it run. A CPU run produces seven rows of fiction.
- Do not add dependencies, do not pip install anything, do not edit
  pyproject.toml or uv.lock (D39). If something is missing, say so.
- Two things must be listened to, not just run: the recording (Level 1) and the
  per-utterance WAVs in utts/ (Level 2). Tell me when it is time and I will
  listen.
- There are three unknowns to record as facts on the first run: whether uv sync
  agrees with the committed lockfile, what defaultSampleRate the loopback
  endpoint reports, and whether it negotiates paInt16 or falls back to
  paFloat32 (D49).

I will be at the machine. Ask me before anything long-running starts, and tell
me what you need me to do (play audio, open a browser, listen to a file).

Finish by updating docs/BENCHMARK.md (survey + results table + the plain
statements it asks for), docs/PLAN.md, docs/STATE.md, docs/DECISIONS.md if the
method changed, and docs/journey/08-*.md — the journey doc is a graded
deliverable. Then commit and push to `build`.
```

---

## What this session is worth

One 10-minute recording unblocks four separate things. That is why this sitting
matters more than its length suggests.

| Blocked since | What | Needs |
|---|---|---|
| session 5 | Level 0's Windows half | `uv sync` + `doctor` — no recording needed |
| session 6 | Level 1's playback check | the recording, listened to |
| session 7 | Level 2's boundary inspection | `dump_utterances --write-wav` over it |
| session 8 | **Level 3's entire measurement** | that recording as `--input-wav` |

Budget: about 30 minutes for Levels 0–2, plus ~6–7 GB of model downloads, plus
~75 minutes for the benchmark matrix. It does not all have to happen in one
sitting, but Levels 0–2 must precede Level 3.

---

## Step 0 — Preflight (Level 0, owed since session 5)

```powershell
git pull
uv sync
python -m speech_translator.doctor
```

`doctor` runs 14 checks. **On this machine, healthy looks like:** no red lines
except the Translation API key if you have not created `.env`, and one WARN for
`Whisper model selected` — which is correct until Level 3 finishes (D44).

Specifically, these three flip from FAIL to PASS here and are the point of
running it:

- `CUDA visible to CTranslate2` → `1 device(s)`
- `cuBLAS / cuDNN loadable` → the DLL directories it added
- `WASAPI loopback device` → the endpoint name, its `defaultSampleRate` and
  channel count

If `cuBLAS / cuDNN loadable` fails, see **Troubleshooting → the DLL trap** below
before going further. Nothing downstream will work until it passes.

**Record:** whether `uv sync` agreed with the committed lockfile or resolved
anything differently. That is the first real test of D39's cross-platform claim,
and it has only ever been argued, never demonstrated.

### The API key, if you want gate 3 to produce a number

```powershell
copy .env.example .env
notepad .env      # paste GOOGLE_TRANSLATE_API_KEY
```

Without it the benchmark still runs, but `MT_roundtrip` is recorded as **missing**
and gate 3 reports `unknown` for every row — deliberately, because substituting
a guessed 300 ms into a gate is what D51 exists to stop. The whole matrix would
then have to be re-judged later. Ten probe calls cost about 180 characters
against the 500 000/month free tier (D31), so this is worth doing first.

---

## Step 1 — Level 1: the capture

```powershell
python -m speech_translator.tools.list_devices
```

**Record:** the endpoint name, index, `defaultSampleRate` and channel count.
Never hard-code any of them — the source reads them off the device (D41, D49).

Then start playing **10+ minutes of natural, meeting-like speech in a
non-English language** through the speakers — a Spanish interview or podcast is
what `BENCHMARK.md` step 2 suggests. Meeting-like, not studio-clean: room tone
and a music bed are what Silero is in the design to survive (D23).

```powershell
python -m speech_translator.tools.record_loopback -t 600
```

Writes `var/recordings/loopback-<timestamp>.wav`, 16 kHz mono int16 — the
pipeline's own frame format, which is what makes the file itself the evidence
for the format layer (D47). Watch the meter; if it reports **"WARNING: every
sample is zero"**, nothing was playing or the wrong endpoint was captured.

**Then listen to the file.** Correct pitch, no clicks at chunk boundaries. That
is Level 1's last open acceptance criterion and no test can close it — the
per-chunk-resampler failure it is really about was measured on Linux at 30.2 dB
of out-of-band energy versus 81.4 dB streaming (D46), but the device itself has
never been opened.

**Record:** whether the endpoint negotiated `paInt16` or fell back to
`paFloat32` (D49) — the tool prints the source description — and the drift line.

---

## Step 2 — Level 2: the boundaries

```powershell
python -m speech_translator.tools.dump_utterances --input-wav var\recordings\loopback-<timestamp>.wav --write-wav utts\
```

**Then listen to the files in `utts\`.** Do the cuts land at real pauses, or do
they chop mid-word? That is Level 2's one open criterion. A table of millisecond
offsets tells you the state machine is self-consistent, not that it cut where a
person would have.

Also check the summary line's VAD cost — it should be near 0.116–0.141 ms per
32 ms frame (D35, D53).

If the boundaries are visibly wrong, **do not silently retune**. The thresholds
(0.50 open / 0.35 release) and the padding (128 ms / 192 ms) are labelled
TUNABLE and were chosen against generated speech (D54, D55); a retune against
real audio is legitimate but it is a Level 7 activity, it changes the chunk
distribution the benchmark measures, and it needs its own DECISIONS entry.
Record what you saw and raise it.

---

## Step 3 — Level 3: the benchmark

### 3a. Pre-download the weights

Otherwise the first configuration's `load_s` is mostly a download, and the
matrix stalls unpredictably between rows. One download per **model size** — the
two `compute_type` variants quantise the same weights at load time, so it is
four downloads, not seven.

```powershell
python -c "from speech_translator.windows_cuda import add_cuda_dll_directories as a; a(); from faster_whisper.utils import download_model; [download_model(m, cache_dir='models') for m in ('small','medium','large-v3-turbo','large-v3')]"
```

About 6–7 GB. `models/` is gitignored and is the cache directory `config.py`
pins (D41).

### 3b. Open a browser

Gate 2 is not "does the model fit in 4 GB" — it is "does it fit **alongside the
desktop compositor and the browser rendering our own captions**" (D18). The demo
has a browser open, so the benchmark does too. Open a few tabs; leave it there
for the whole run.

### 3c. Run it

```powershell
python -m speech_translator.tools.benchmark --input-wav var\recordings\loopback-<timestamp>.wav --language es
```

Defaults are the ones `BENCHMARK.md` specifies: the seven configurations, 10
**wall-clock** minutes each with the audio looped (D59), chunks cut from the real
`Segmenter` once and shared byte-identically across all rows (D57).
`--language` must match the recording.

What it does before it times anything:

1. Refuses to start unless `get_cuda_device_count()` returns 1 (D36).
2. Times ten live translation calls and takes the p95 (D51).
3. Cuts the chunk list and prints the distribution — check that
   `count`, `p50` and `p95` look like speech and not like silence.

Then, per configuration, it prints a line and writes into
`var\benchmark\<timestamp>\`:

| file | what |
|---|---|
| `results.json` | the whole run — the machine-readable artefact |
| `results.md` | `BENCHMARK.md`'s table, rendered, **ready to transcribe** |
| `chunks-*.jsonl` | per-chunk timing, appended live |
| `gpu.jsonl` | `nvidia-smi` at 1 Hz |

**Everything is written incrementally.** A kill at minute 55 keeps minutes 0–54,
and a configuration that OOMs is recorded as a failed row while the matrix
continues. Do not restart the matrix from the top because one row died.

Take the screenshots `BENCHMARK.md` asks for while it runs: `nvidia-smi`
mid-run, and the throttling curve.

### 3d. The bracket, for the selected row only

Once a row looks like the winner, re-run just that one at fixed chunk lengths —
D52's bracket, kept alive by D57:

```powershell
python -m speech_translator.tools.benchmark --input-wav <same wav> --language es --configs <model>:<compute_type> --chunking fixed --chunk-ms 4000
python -m speech_translator.tools.benchmark --input-wav <same wav> --language es --configs <model>:<compute_type> --chunking fixed --chunk-ms 1500
```

If a bracket fails a gate the headline run passes, **that is the finding** and it
goes in the report. About 20 extra minutes.

### 3e. Choose the model

Read `results.md`, then read the `samples` field in `results.json` — real
transcripts in the source language, kept for exactly this. **Listen** to a few
of the corresponding utterances in `utts\` and judge whether the text is good
enough.

The rule is **the fastest that is accurate enough, not the largest that fits**
(D25/D26). RTF pays twice — it multiplies `D` in the word-age formula — so a
model at RTF 0.25 beats one at RTF 0.49 even though the first two columns call
both a pass. Select on the word-age column, which is labelled **predicted** and
must stay labelled that way.

### 3f. Write the choice into config

Two places in `speech_translator/config.py`, and they must agree:

```python
    model_size: str | None = None          # -> "<chosen>"
    compute_type: str | None = None        # -> "<chosen>"
```

```python
        model_size=_env_opt_str("MODEL_SIZE", None),      # -> default "<chosen>"
        compute_type=_env_opt_str("COMPUTE_TYPE", None),  # -> default "<chosen>"
```

Into `config.py`, not `.env`: this is a project decision that the Linux box needs
to see too. `.env` stays for per-machine overrides.

Then `python -m speech_translator.doctor` should flip `Whisper model selected`
from WARN to PASS, and `Whisper weights cached` from WARN to PASS. That is the
cheapest possible confirmation that the value landed where the code reads it.

---

## Troubleshooting

**The DLL trap (D43) — `import ctranslate2` fails with "DLL load failed".** This
is the single most likely thing to eat the session, and the error names the wrong
component. The cuBLAS/cuDNN wheels unpack to `site-packages\nvidia\<component>\bin`
and nothing puts those on the Windows DLL search path;
`windows_cuda.add_cuda_dll_directories()` does, and it must run **before** the
first `ctranslate2` import. `doctor` and the harness both call it. If the check
still fails: confirm `uv sync` installed `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12`, and that `doctor --json` lists the directories it added.
Do not hand-copy DLLs — that is what D41 rejected, because it is not reproducible.

**`get_cuda_device_count()` returns 0.** Stop. Do not pass `--device cpu` to make
it run. Check the driver, then the DLL trap above. There is no CPU fallback in
this project (D36) and a CPU run would produce a full table of numbers describing
a machine we are not shipping on.

**A configuration OOMs.** Expected, and handled — it is a failed row and the run
continues. `large-v3` at `float16` is not even in the matrix for this reason
(~3.1 GB of weights on a 4 GB card that is also drawing the desktop). If
*nothing* clears the bar, `BENCHMARK.md`'s fallback ladder applies in order:
`int8_float16`, then a smaller model, then reduce `max_utterance_ms` (floor
2000 ms, D28 — and note that this *helps* gate 3 while hurting translation
quality). Reducing `max_utterance_ms` is a real decision and needs a DECISIONS
entry, not a quiet config edit.

**"no chunks from ...: the Segmenter found no speech."** The recording is silent
or the wrong endpoint was captured. Check it with `dump_utterances` before
spending 70 minutes on it — which is what the harness tells you to do.

**The console mangles a caption.** UTF-8 is forced at package import (D42), so
this should not happen; if it does, it is a real bug and worth recording, because
the Windows console being cp1252 is the D41 register's most likely
"works on my machine" failure and this is the first time it has been on a real
Windows console.

**Something takes longer than expected.** The matrix is ~75 minutes plus
downloads. `--minutes` shortens a run, but anything under 10 stops being the
sustained test the throttling column depends on — record it as a deviation if you
do it.

---

## What to update afterwards

### `docs/BENCHMARK.md` — the primary deliverable of this level

- **Hardware under test** table: driver / CUDA version, **VRAM in use at idle**
  (it comes off the budget), CPU / RAM. From `nvidia-smi` and the machine, not
  estimated.
- **Results table**: transcribe from `var\benchmark\<stamp>\results.md`. Do not
  retype the numbers by hand — that is why the harness emits the table in
  `BENCHMARK.md`'s own columns.
- The subjective **accuracy note** for the selected configuration, in the demo's
  source language.
- Then the four plain statements the file asks for:
  - selected model size and `compute_type`, with the RTF, peak VRAM **and** p95
    word age that justify them;
  - the **throttling penalty** between first and last minute;
  - whether `int8_float16` actually beat `float16` on a part with no tensor
    cores (D18 says do not assume the RTX answer transfers);
  - whether any headroom remains for the diarization stretch goal (D9/D21 —
    expect no, and say so with the number).
- Change the file's **Status** line from "not yet run".

### `speech_translator/config.py`

`model_size` and `compute_type`, in both places (Step 3f).

### `docs/PLAN.md`

Tick the boxes that were earned, each with a one-line italic note saying what the
evidence was — the style Level 1's third box already uses. Candidates:

- Level 0: `uv sync` on Windows, `doctor` on Windows.
- Level 1: the 10-minute capture that plays back cleanly; `list_devices` against
  real hardware; the `paInt16`/`paFloat32` answer.
- Level 2: boundaries land at real pauses on a captured meeting WAV.
- Level 3: all five, if they were met. Leave anything unmet unticked with a note
  saying why — that is more useful than a tick.

Update the status column for every level that moved.

### `docs/STATE.md`

- The date and session number in the header.
- Move the closed items out of **Blocked** — in particular delete the
  four-row "one missing recording blocks four things" table, which is the whole
  point of this session.
- Add what was learned under **Done**: the three recorded facts (lockfile
  agreement, `defaultSampleRate`, sample format), the benchmark's headline
  numbers, the selected model.
- Update the level table and **Next** (which becomes Level 4 — Transcription).
- Close the open question "whether `int8_float16` beats `float16` on a GPU with
  no tensor cores" with the measured answer.

### `docs/DECISIONS.md`

Only if the method changed. Append, never edit an existing entry — this project
supersedes by appending (D12 under D25, D6 under D16). Likely candidates:

- the fallback ladder was needed (a smaller model, or `max_utterance_ms`
  reduced);
- the D54/D55 VAD values were retuned against real audio;
- something about this machine turned out not to match what the docs assumed.

### `docs/journey/08-*.md`

The graded deliverable. It wants: the prompt used, the raw numbers, the
throttling curve, the reasoning that turned numbers into a model choice, the
screenshots, and anything that went wrong — **a negative result documented
honestly is worth more than an untested assumption that happened to hold.** If
the DLL trap bit, that is the best material in the session; write down what
actually fixed it.

### Commit and push

```powershell
git add -A
git commit
git push origin build
```

Message in the existing style: a subject line naming the level and the decisions,
then the reasoning. The Linux box picks it up from there.

---

## Definition of done for this session

- [ ] `uv sync` and `doctor` have run on this machine, and `doctor` shows no
      unexpected red lines.
- [ ] A 10-minute recording exists, has been listened to, and plays back cleanly.
- [ ] `utts\` has been listened to and the boundaries judged.
- [ ] The benchmark has run to completion, or every row that failed is recorded
      as a failed row with its reason.
- [ ] `BENCHMARK.md`'s survey and results table are filled from observation.
- [ ] A model is selected on the word-age column, with a listening judgement
      behind "accurate enough", and written into `config.py`.
- [ ] `PLAN.md`, `STATE.md` and `docs/journey/08-*.md` reflect what happened.
- [ ] Pushed to `build`.

## Do not

- Do not fabricate, estimate or interpolate a single cell of the results table.
- Do not tick an acceptance box that was not earned; an unticked box with an
  honest note is a better artefact than a tick.
- Do not let the harness choose the model. It reports; the choice is human and
  needs a listening test.
- Do not run the benchmark on CPU to "get some numbers".
- Do not add dependencies or edit `uv.lock` (D39).
- Do not hard-code a sample rate, a device index, or a model size anywhere
  outside `config.py`.
