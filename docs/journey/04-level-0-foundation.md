# Journey 04 — Level 0, and the DLLs that were installed but not loadable (2026-08-18)

Session with Claude (Opus 5) in Claude Code. First session that wrote code. Four
sessions of design, then one session that turned it into a repo — which is a
ratio I want to defend rather than apologise for, and the defence is in "What I
learned" at the bottom.

## The prompt that started it

> New session on the speech translator. Design phase is complete; we're starting
> implementation. CLAUDE.md loads automatically — read these in order before
> doing anything:
>
> 1. `docs/STATE.md` — where we are (session 4 done, no application code yet)
> 2. `docs/PLAN.md` — the build ladder, levels 0-8. We are starting Level 0.
> 3. `docs/INTERFACES.md` — contracts for the stages Level 0/1 touch
> 4. `docs/DECISIONS.md` — D25, D27, D35, D39, D40, D41 are the ones that bind
>    Level 0. Skim the rest; do not re-litigate them.
>
> Task: build Level 0 (Foundation), per `docs/PLAN.md`. Deliverables are
> `pyproject.toml` + committed `uv.lock`, the `speech_translator` package
> skeleton, `config.py` holding the D25 values, vendored
> `assets/silero_vad.onnx`, `.env.example`, `.gitattributes`, forced UTF-8, and
> `python -m speech_translator.doctor`.
>
> Constraints I don't want re-derived:
> • Never add torch/torchaudio. Silero runs from the vendored ONNX via
>   onnxruntime — verified at 0.156 ms per 32 ms frame (D35).
> • uv + uv.lock, not pip/requirements.txt (D39). uv 0.11.7 is installed.
> • Package must import cleanly on Linux; Windows-only imports are lazy (D22).
> • model_size and compute_type stay None in config — the benchmark decides them
>   at Level 3 (D20).
>
> I develop on Linux and test on Windows via git, so `doctor` reporting CUDA and
> loopback as FAIL on this Linux machine is correct output, not a bug.
>
> Finish by running Level 0's acceptance criteria from PLAN.md and telling me
> which pass. […] Still blocked, unchanged: the truncated scope line in the
> assignment brief (PRD.md) and the deadline. Don't design around them, just
> don't forget them.

The "do not re-litigate" line was deliberate. Sessions 1–4 produced 41 decisions
and I did not want to spend session 5 re-deciding any of them; I wanted them
spent.

## What got built

`pyproject.toml` + `uv.lock` (45 packages, zero torch), the package skeleton with
`config.py`, `logging_setup.py`, `windows_cuda.py`, `doctor.py` and a
`__main__.py` stub, `assets/silero_vad.onnx` with its MIT license and a
provenance note, `.env.example`, `.gitattributes`, and 16 tests that pass on
Linux with no GPU, no audio hardware and no network.

`doctor` runs 14 checks. On this laptop: 7 pass, 3 expected-fail, 1 warn, 1 real
fail (I have not made the Google API key yet).

```
speech-translator doctor · Linux x86_64 · target platform: Windows (D16)

 [PASS]  Python 3.12                  3.12.3 (linux-x86_64)
 [PASS]  UTF-8 output streams         stdout/stderr utf-8 · non-ASCII renders: «señor» «こんにちは»
 [PASS]  Versions match uv.lock       40 packages match (45 locked)
 [PASS]  No torch in the environment  torch, torchaudio absent
 [PASS]  Latency budget (D25)         silence=600 max_utt=4000 min=300 floor=2000 queue=4 → predicted p95 word age at RTF 0.5: 6.9 s (target < 7 s)
 [PASS]  Silero VAD (vendored ONNX)   2.3 MB · 0.110 ms per 32 ms frame (D35 measured 0.156)
 [FAIL]  CUDA visible to CTranslate2  get_cuda_device_count() == 0 — no CUDA device (no CPU fallback is built, D36)  (expected here — Windows is the target)
 [FAIL]  cuBLAS / cuDNN loadable      Windows-only check (the DLL trap does not exist on Linux)  (expected here — Windows is the target)
 [WARN]  Whisper model selected       model_size/compute_type unset — decided by docs/BENCHMARK.md at Level 3 (D20). Expected until that runs.
 [SKIP]  Whisper weights cached       no model selected yet · cache: /home/tekioniot/Desktop/speech_translator/models
 [FAIL]  WASAPI loopback device       pyaudiowpatch unavailable (ModuleNotFoundError) — Windows-only dependency, marked sys_platform == 'win32'  (expected here — Windows is the target)
 [PASS]  State directory writable     /home/tekioniot/Desktop/speech_translator/var
 [FAIL]  Translation API key          GOOGLE_TRANSLATE_API_KEY unset — copy .env.example to .env (D37)
 [SKIP]  Live translation             --no-net

7 passed · 1 failed · 3 expected-fail on this platform · 1 warning
Linux dev box: CUDA and loopback cannot pass here by design (D17, D22). Run this again on the Windows machine after every git pull.
Fix the failed checks above before running the pipeline.
```

## The finding — D41's mitigation was incomplete

The best thing to come out of the session was not in the deliverables list.

D41 says to take cuBLAS and cuDNN from the `nvidia-cublas-cu12` /
`nvidia-cudnn-cu12` pip wheels rather than hand-copying DLLs, because wheels are
pinned by the lockfile. While writing the `doctor` check for that, the model
noticed that installing the wheels **does not make the DLLs loadable**: they
unpack to `site-packages/nvidia/<component>/bin`, and nothing puts those
directories on the Windows DLL search path. PyTorch does that for its users. We
deliberately have no PyTorch (D35), and CTranslate2 does not do it.

The failure this would have produced on the demo machine:

```
ImportError: DLL load failed while importing translator: The specified module could not be found.
```

Pointing at CTranslate2. Not at cuDNN. Not at the wheel. On a Windows box, over
git, at Level 3 — the benchmark — which is already the highest-risk level on the
ladder.

The fix is `speech_translator/windows_cuda.py`: walk the installed `nvidia`
namespace package, call `os.add_dll_directory()` for each `bin` directory that
holds DLLs, no-op on Linux. It is now called before the first `import
ctranslate2` in `doctor`, and it has to be called again in the worker process at
Level 4, because on Windows that process is spawned and re-imports everything
cold (D27). Recorded as **D43**, specifically because the fix is invisible: delete
it and everything still passes on Linux.

This is the same shape as session 3's torch discovery. A mitigation written from
general knowledge was one step short of working, and the missing step was only
visible to someone actually looking at where the files land.

## What did not go smoothly

**The first test run died before collecting a single test.** This laptop exports
`PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` globally, pytest
autoloads a ROS plugin from it, and that plugin imports `lark`, which is not
installed. Nothing to do with the project.

The recovery is the part worth recording. The obvious fix is a line in
`pyproject.toml` disabling that plugin, and I would probably have taken it. The
model declined, on the grounds that a ROS-specific opt-out in the project's
config would outlive the reason for it and confuse whoever reads it next —
including me on the Windows machine, where ROS does not exist. It ran
`PYTHONPATH= uv run pytest` instead and wrote the quirk into `STATE.md` under a
"dev box only" heading. Right call. It also cost about ninety seconds, which is
the entire argument for making it: the workaround is cheap and permanent
project-file pollution is not.

**Two judgement calls I want to flag rather than let pass silently:**

1. `config.py` now holds `no_speech_prob_max = 0.6` and `avg_logprob_min = -1.0`
   for D30's hallucination guard. D30 specifies the *rule* but no numbers, so
   these are starting points, not derived values — and this project has spent
   two sessions insisting on the difference. They are commented `TUNABLE, not
   derived` directly above, and they are calibrated at Level 4/7 against real
   transcripts. I would rather have them visible with a warning label than
   invented at the point of use.
2. `doctor` exits 1 when the Google API key is missing. That is honest, but it
   means a clean dev box cannot get a green run until I create the key. I am
   leaving it as a failure — an absent credential is genuinely absent — but it
   is a deliberate choice, not an oversight.

## My assessment of the output

**What I rate highly.** It read the four docs in the order I asked and then
worked from them rather than from its own defaults: `model_size` and
`compute_type` really are `None`, the D25 values are in `config.py` with the
derivation next to them, and nothing tried to helpfully suggest `small` +
`int8_float16` "just to get started". That specific temptation is the one that
would have quietly voided Level 3.

It also copied the vendored ONNX out of `site-packages` **before** running
`uv sync`, because `uv sync` prunes `silero-vad` from the venv and would have
deleted the file it was about to vendor. I did not think of that ordering and
would have hit it.

**What I had to supply.** Almost nothing this session, which is a direct result
of the four design sessions. That is the return on them: the prompt was fourteen
lines and the model did not need to ask a single clarifying question, because
`PLAN.md` already said what "done" meant and `DECISIONS.md` already said why.

**Where it fell short.** The cuDNN gap was caught while writing the check, not
while writing the dependency list — the first pass added the nvidia wheels to
`pyproject.toml` and moved on, exactly as D41 described, and it was only when it
sat down to *verify* the mitigation that the hole appeared. That is an argument
for `doctor` existing (D40) more than it is a criticism, but it is worth being
precise about: writing the check found the bug; writing the dependency did not.

**Where I would push back.** `doctor` now has 14 checks and its own JSON output
mode, for a project with no pipeline yet. On a different project I would call
that gold-plating. Here D40 argued the case a session in advance — every item on
it fails on the *other* machine, hours later, pointing at the wrong layer — and
the cuDNN find justified the whole thing before the first `git push`.

## Decisions taken

| | Chosen | The reason that survives |
|---|---|---|
| D42 | `force_utf8()` at package import, not per entry point | the entry point that forgets is the spawned worker, on Windows, at Level 4 |
| D43 | Register the wheel CUDA DLL dirs with `os.add_dll_directory()` | installed ≠ loadable; PyTorch does this and we removed PyTorch |
| D44 | `doctor` grades failures by whether this platform *can* pass | red on Linux + exit 0; red on Windows + exit 1 |
| D45 | `required-environments` + `requires-python` bound both ends | a Linux-only resolution must fail at lock time, not on the demo machine |

## Level 0 acceptance — what actually passes

| Criterion | Result |
|---|---|
| `uv sync` on Linux and Windows from one lockfile | Linux **verified**. Windows **owed** — all 45 packages carry a `win_amd64`/any wheel or an sdist, checked programmatically, but that is evidence and not the run |
| `import speech_translator` on Linux | **pass** |
| `doctor` runs and reports honestly on both | Linux **pass** (CUDA + loopback FAIL, as designed). Windows **owed** |
| `pip list \| grep torch` returns nothing | **pass** — and the string `torch` does not appear in `uv.lock` at all |

Three of four fully pass; the remaining half of two of them needs the Windows
machine and is written into `STATE.md` as the first thing to do there.

## What I learned

**Four design sessions bought a fourteen-line prompt.** The thing I was most
worried about — that all that documentation was procrastination — is answerable
now. The implementation session asked me zero questions and re-opened zero
decisions, and the two places it had to choose something (the guard thresholds,
the doctor exit code) it flagged rather than buried, because the project had
already established that derived and guessed are different words.

**"It's installed" and "it loads" are different claims, and only one of them is
testable from here.** That is now the second time on this project that a
dependency was in the tree in a state that would not work — `silero-vad` in
session 3, cuDNN this session. Both were found by looking at the actual files
rather than at the dependency list. `doctor` is the mechanism that makes that
look happen at `git pull` time, on the machine where it matters.

**A red line that is expected is not the same as a red line that is wrong**, and a
tool that cannot tell them apart stops being read. That is the whole content of
D44 and it is a one-boolean feature.

## Next

Level 1 — audio capture. But first, on the Windows machine: `git pull`,
`uv sync`, `python -m speech_translator.doctor`. That closes Level 0 and it is
also the first real test of D39's claim that one lockfile serves both machines.

---

**Screenshots to capture next session:** `doctor` on Linux (this session's
output, showing the labelled expected-failures), `doctor` on Windows with CUDA
and loopback green, and the two side by side — that pair is the clearest single
image of what D22's dev/test split actually means.
