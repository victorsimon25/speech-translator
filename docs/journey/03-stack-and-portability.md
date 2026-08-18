# Journey 03 — Stack confirmation, and the torch that was hiding in the VAD (2026-08-18)

Session with Claude (Opus 5) in Claude Code. Chat only again, by intent. The goal
was to stop guessing at portability *after* the code existed.

## The prompt that started it

> is there anything more to discuss? and are we sure on the stacks also i wana
> make sure nowitslf that we shoudnt have problems like running in my system but
> not in others

Three questions in one sentence, and the third is the one that mattered. My
concern was the Linux-write / Windows-run split (D22) — code that imports fine on
the laptop and dies on the machine that has to demo it.

## The finding

Rather than answering from memory, the model went and read the installed
`silero-vad` package. What it found:

```python
# silero_vad/model.py, line 2
import torch
```

Unconditional, at module scope — including on the `onnx=True` path that exists
precisely so you don't need torch. The package declares `torch` and `torchaudio`
as hard requirements. It also demonstrated that `import silero_vad` **already
fails in my venv**, because torch was never installed. That import has been
broken since the day I created the environment and I had not noticed, because no
code imports it yet.

The consequence chain is the part I would not have traced in time:

1. On Windows, `pip install silero-vad` pulls torch — ~2.5 GB, possibly a CUDA
   build.
2. A torch CUDA runtime and CTranslate2's CUDA runtime in one process means two
   CUDA contexts and two sets of cuDNN expectations.
3. The card has **4 GB total**, and D18 already established that VRAM — not
   throughput — is the binding constraint.
4. It would work perfectly on the Linux dev box, where nothing ever loads CUDA.

That is the exact failure I asked about, sitting in the dependency list, and it
would have surfaced as a CUDA OOM or a DLL error on the Windows machine with the
error message pointing at Whisper rather than at the VAD.

## The fix, and the measurement that came free

The package ships the ONNX weights as data files. Load them with `onnxruntime`
directly, vendor the 2.3 MB file into the repo (Silero VAD is MIT), and never
import the package's Python at all. torch and torchaudio leave the dependency
tree entirely.

Then it benchmarked the thing it had just proposed, unprompted:

```
model  : silero_vad.onnx, 2.3 MB
inputs : input [N,512] · state [2,N,128] · sr
cost   : 0.156 ms per 32 ms frame
```

D23 estimated "roughly 1 ms of CPU per 32 ms frame". The real number is ~6×
cheaper — 0.5% of one core rather than 3%. Small in absolute terms, but it is
the second time on this project that a measured number has replaced a reasonable
guess, and both times the guess was the pessimistic one.

## My assessment of the output

**What I rate highly:** it treated "are we sure on the stack" as a question to be
answered with evidence from the machine rather than an invitation to restate the
stack approvingly. Reading the installed package's source is an obvious move in
hindsight and I did not think to ask for it.

**What I had to supply:** the scoping. Its portability analysis initially covered
"other people's machines" in general, which is a much bigger problem than the one
I have. Once I confirmed only my Windows box has to run it, a CPU fallback path —
which would have reopened D6 and doubled the benchmark — disappeared from the
plan. The lesson is the same one from session 1: the model will answer the
question you asked, and "make it portable" is a wider question than "make it run
on my other machine".

**Where I disagreed:** none this session. The four choices it costed out
(CUDA-only, REST + API key, FastAPI, uv lockfile) were the ones I would have
picked, but I would not have had the reasons written down.

## Decisions taken

| | Chosen | The reason that survives |
|---|---|---|
| D35 | Vendored Silero ONNX, no torch | two CUDA runtimes on a 4 GB card |
| D36 | Demo machine only, no CPU fallback | the brief says "a UI you can demonstrate", not one they install |
| D37 | Translation over REST + API key | no absolute credential path to differ across machines |
| D38 | FastAPI + uvicorn, vanilla JS | no node toolchain between a change and a demo |
| D39 | `uv` + committed `uv.lock` | a flat requirements file pins direct deps, not the resolved graph |
| D40 | `doctor` preflight, built in step 1 | turns "it fails on Windows" into a line item |
| D41 | Portability register | including that the Windows console is cp1252 and will crash on printing a Spanish caption |

D41's cp1252 entry is the one I want to flag for anyone reading this later: the
app's entire purpose is producing non-ASCII text, and the default Windows console
encoding cannot print it. That is a crash in the demo, on the happy path.

## What I learned

Asking "will this be portable?" before writing code is worth more than it sounds,
but only if the answer comes from inspecting the actual dependency tree rather
than from general good practice. The generic advice — pin versions, use env vars,
watch your paths — would not have found `import torch` inside a VAD package I had
already decided was cheap and safe.

The second thing: **an unused dependency is an untested dependency.** `silero-vad`
had been in the venv for a session and a half in a state where importing it
raised. Nothing caught that because nothing imported it. The `doctor` command
(D40) exists so that gap closes at `git pull` time rather than at demo time.

## Next

Build step 1: project skeleton (`pyproject.toml` + `uv.lock`, vendored ONNX,
config), `AudioSource` (WASAPI loopback + WAV), the two capture tools, and
`doctor`.

---

**Screenshots to capture next session:** `doctor` output on the Windows machine
with everything green, `list_devices` showing the WASAPI loopback device, and a
captured 16 kHz mono WAV in a waveform viewer.
