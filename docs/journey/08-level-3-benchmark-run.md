# Session 9 — Level 3 benchmark run

_2026-08-20. Windows demo machine (GTX 1650, 4 GB). Claude Code session._

## What happened

Levels 0-2 were closed on Windows in an earlier Gemini session (2026-08-19):
`uv sync`, `doctor`, 10-minute recording, 160 utterances segmented. Five
benchmark attempts then failed with `RuntimeError: Library cublas64_12.dll is
not found or cannot be loaded`.

This session fixed the DLL issue (D60) and ran the full 7-configuration
benchmark to completion.

## The DLL trap (D43/D60)

`os.add_dll_directory()` registers directories with Python's own restricted
DLL loader. CTranslate2's native code loads cuBLAS via `LoadLibrary` with a
bare filename, which searches a different path: app dir, system32, Windows,
current dir, then PATH. The fix: also prepend the nvidia wheel bin directories
to `os.environ["PATH"]`.

Evidence the fix was needed: doctor passed, `get_cuda_device_count()` returned
1, model loaded (3.6 s), but the first transcription call died. The cuBLAS
library is not needed for device enumeration or weight loading — only for the
GEMM calls during inference.

## Benchmark results

| Model | compute_type | RTF | Peak VRAM | Gate 1 | Throttling |
|---|---|---|---|---|---|
| small | float16 | 0.652 | 991 MB | FAIL | +0.2% |
| **small** | **int8_float16** | **0.279** | **711 MB** | **PASS** | -2.0% |
| medium | float16 | 1.580 | 2801 MB | FAIL | +8.7% |
| **medium** | **int8_float16** | **0.488** | **1849 MB** | **PASS** | -10.6% |
| large-v3-turbo | float16 | 2.622 | 2369 MB | FAIL | +30.4% |
| large-v3-turbo | int8_float16 | 0.606 | 1353 MB | FAIL | -6.3% |
| large-v3 | int8_float16 | 0.718 | 3217 MB | FAIL | -7.4% |

Gate 3 (word age): unknown — no API key present. At assumed 300 ms MT, medium
predicts 6852 ms (< 7000 ms gate).

## The int8 answer (D18's open question)

**int8_float16 dramatically outperforms float16 on this GPU with no tensor
cores.** The speedup ratios: 2.3x (small), 3.2x (medium), 4.3x (turbo).

This is not the "tensor core" story. The GTX 1650's TU117 has no INT8 tensor
cores. The speedup comes from **memory bandwidth**: int8 weights are half the
size, so they transfer from VRAM to the compute units in half the time. On a
bandwidth-limited part (128-bit bus, 192 GB/s), halving the data moved halves
the time spent waiting for it.

The float16 configs show positive throttling (up to +30.4% for turbo), while
int8_float16 configs show negative throttling. This confirms: float16 is
thermally stressing the card enough to trigger clock reduction; int8 is not.

## Model selection

**medium:int8_float16.** The rule is "fastest that is accurate enough" (D25/D26).

- small:int8_float16 is faster (RTF 0.279) but garbles proper nouns and
  code-switches on bilingual audio. Feeding "Spanner Soccerstar Lameño Mál"
  to the translator produces garbage.
- medium:int8_float16 (RTF 0.488) produces coherent Spanish — correct grammar,
  consistent language, proper vocabulary.

The RTF margin is thin (0.488 vs gate of 0.5) but sustainable: negative
throttling means it gets faster over time, not slower. First-minute RTF of
0.530 drops to 0.474 in the last minute.

## What went wrong

1. The DLL trap ate 5 failed attempts in the previous session before being
   understood. `doctor` passing was misleading because it loads DLLs by
   absolute path (`ctypes.WinDLL(str(dll))`), which always works regardless
   of the search path.
2. No Google Translate API key meant gate 3 could not be fully evaluated.
   The prediction passes at assumed latency but the measured value is owed.
3. The recording's chunk distribution is 57% max-length cuts (91/160), meaning
   most utterances hit the 4 s cap rather than closing on silence. This may
   indicate the VAD thresholds (D54/D55) need tuning for this audio — a Level 7
   activity.

## Decisions made

- **D60:** PATH prepend alongside `os.add_dll_directory()` for CUDA DLL loading.
- **Model choice:** medium:int8_float16 written into `config.py`.
