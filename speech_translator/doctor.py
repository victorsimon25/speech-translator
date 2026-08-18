"""Preflight check: ``python -m speech_translator.doctor``.

Every risk on the D41 portability register fails on the *other* machine, hours
after the code was written, with an error message pointing at the wrong layer.
This command turns each of them into a line that names the missing thing (D40).
Run it after every ``git pull`` on the Windows machine — that is the actual dev
loop (D22).

**On Linux, the CUDA and loopback checks report FAIL and that is correct
output, not a bug.** Windows is the target platform (D16, D17); the Linux box
writes the code. Those failures are marked "expected here" and do not affect
the exit code, so a red line on the dev laptop stays distinguishable from a red
line on the demo machine.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import sysconfig
from dataclasses import dataclass, field
from typing import Callable

from . import config as cfg
from .logging_setup import force_utf8, stream_is_utf8
from .windows_cuda import add_cuda_dll_directories, cuda_dll_dirs

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

IS_WINDOWS = sys.platform == "win32"


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    #: True when this check can only pass on the Windows demo machine, so a
    #: failure on the Linux dev box is expected and not counted against us.
    windows_only: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def expected_failure(self) -> bool:
        return self.status == FAIL and self.windows_only and not IS_WINDOWS


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_python() -> Result:
    v = sys.version_info
    detail = f"{platform.python_version()} ({sysconfig.get_platform()})"
    ok = (v.major, v.minor) == (3, 12)
    return Result("Python 3.12", PASS if ok else FAIL, detail)


def check_utf8() -> Result:
    """D41: the Windows console is cp1252 and every caption is non-ASCII."""
    force_utf8()
    streams = {"stdout": sys.stdout, "stderr": sys.stderr}
    bad = [n for n, s in streams.items() if not stream_is_utf8(s)]
    if bad:
        return Result("UTF-8 output streams", FAIL, f"not UTF-8: {', '.join(bad)}")
    return Result(
        "UTF-8 output streams",
        PASS,
        "stdout/stderr utf-8 · non-ASCII renders: «señor» «こんにちは»",
    )


def check_lockfile(_c: cfg.Config) -> Result:
    """Installed versions must match the committed lockfile (D39)."""
    import tomllib
    from importlib.metadata import distributions

    lock_path = cfg.REPO_ROOT / "uv.lock"
    if not lock_path.exists():
        return Result("Versions match uv.lock", FAIL, f"missing {lock_path}")

    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    locked = {p["name"].lower().replace("_", "-"): p.get("version") for p in lock["package"]}

    mismatched: list[str] = []
    matched = 0
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name not in locked or locked[name] is None:
            continue
        if dist.version == locked[name]:
            matched += 1
        else:
            mismatched.append(f"{name} {dist.version} != {locked[name]}")

    if mismatched:
        return Result(
            "Versions match uv.lock",
            FAIL,
            "; ".join(mismatched[:5]) + "  — run: uv sync",
        )
    return Result("Versions match uv.lock", PASS, f"{matched} packages match ({len(locked)} locked)")


def check_no_torch() -> Result:
    """D35: torch anywhere means a second CUDA runtime on a 4 GB card."""
    from importlib.util import find_spec

    found = [m for m in ("torch", "torchaudio") if find_spec(m) is not None]
    if found:
        return Result(
            "No torch in the environment",
            FAIL,
            f"{', '.join(found)} installed — see D35; a second CUDA runtime will "
            f"OOM the 4 GB card",
        )
    return Result("No torch in the environment", PASS, "torch, torchaudio absent")


def check_silero_asset(_c: cfg.Config) -> Result:
    """Vendored ONNX present, loadable, and one real 32 ms frame through it."""
    path = cfg.SILERO_ONNX_PATH
    if not path.exists():
        return Result("Silero VAD (vendored ONNX)", FAIL, f"missing {path}")
    try:
        import time

        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = np.zeros((1, cfg.FRAME_SAMPLES), dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(cfg.SAMPLE_RATE, dtype=np.int64)
        sess.run(None, {"input": x, "state": state, "sr": sr})  # warm up
        t0 = time.perf_counter()
        n = 50
        for _ in range(n):
            out, state = sess.run(None, {"input": x, "state": state, "sr": sr})
        per_frame_ms = (time.perf_counter() - t0) * 1000 / n
    except Exception as exc:  # noqa: BLE001 — doctor reports, never raises
        return Result("Silero VAD (vendored ONNX)", FAIL, f"{type(exc).__name__}: {exc}")

    size_mb = path.stat().st_size / 1e6
    return Result(
        "Silero VAD (vendored ONNX)",
        PASS,
        f"{size_mb:.1f} MB · {per_frame_ms:.3f} ms per {cfg.FRAME_MS} ms frame "
        f"(D35 measured 0.156)",
        extra={"per_frame_ms": round(per_frame_ms, 4)},
    )


def check_cuda() -> Result:
    """CTranslate2 must see exactly the one GPU (BENCHMARK.md's first gate)."""
    add_cuda_dll_directories()  # must happen before the import on Windows (D41)
    try:
        import ctranslate2
    except Exception as exc:  # noqa: BLE001
        return Result("CUDA visible to CTranslate2", FAIL, f"import failed: {exc}", windows_only=True)
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        return Result("CUDA visible to CTranslate2", FAIL, f"{type(exc).__name__}: {exc}", windows_only=True)

    if count < 1:
        return Result(
            "CUDA visible to CTranslate2",
            FAIL,
            "get_cuda_device_count() == 0 — no CUDA device (no CPU fallback is built, D36)",
            windows_only=True,
        )
    return Result("CUDA visible to CTranslate2", PASS, f"{count} device(s)", windows_only=True)


def check_cuda_libs() -> Result:
    """cuBLAS/cuDNN come from pip wheels, not hand-copied DLLs (D41)."""
    if not IS_WINDOWS:
        return Result(
            "cuBLAS / cuDNN loadable",
            FAIL,
            "Windows-only check (the DLL trap does not exist on Linux)",
            windows_only=True,
        )
    dirs = cuda_dll_dirs()
    if not dirs:
        return Result(
            "cuBLAS / cuDNN loadable",
            FAIL,
            "nvidia-cublas-cu12 / nvidia-cudnn-cu12 not installed — run: uv sync",
            windows_only=True,
        )

    add_cuda_dll_directories()
    wanted = [p for d in dirs for p in sorted(d.glob("*.dll")) if p.name.startswith(("cublas64", "cudnn64"))]
    if not wanted:
        return Result(
            "cuBLAS / cuDNN loadable",
            FAIL,
            f"no cublas64*/cudnn64* DLLs under {[str(d) for d in dirs]}",
            windows_only=True,
        )

    failed: list[str] = []
    for dll in wanted:
        try:
            ctypes.WinDLL(str(dll))  # type: ignore[attr-defined]
        except OSError as exc:
            failed.append(f"{dll.name}: {exc}")
    if failed:
        return Result("cuBLAS / cuDNN loadable", FAIL, "; ".join(failed[:3]), windows_only=True)

    loaded_dirs = sorted({str(p.parent) for p in wanted})
    return Result(
        "cuBLAS / cuDNN loadable",
        PASS,
        f"{len(wanted)} DLLs load · dirs added to the loader path: {'; '.join(loaded_dirs)}",
        windows_only=True,
        extra={"dll_dirs": loaded_dirs},
    )


def check_model_choice(c: cfg.Config) -> Result:
    """Level 3 writes these in; until then, blank is the correct value (D20)."""
    if c.model_size is None or c.compute_type is None:
        return Result(
            "Whisper model selected",
            WARN,
            "model_size/compute_type unset — decided by docs/BENCHMARK.md at "
            "Level 3 (D20). Expected until that runs.",
        )
    return Result(
        "Whisper model selected",
        PASS,
        f"{c.model_size} · {c.compute_type} · device={c.device}",
    )


def check_model_cache(c: cfg.Config) -> Result:
    """First run downloads weights; the cache directory is pinned (D41)."""
    cache = c.model_cache_dir
    if c.model_size is None:
        return Result("Whisper weights cached", SKIP, f"no model selected yet · cache: {cache}")
    hits = list(cache.glob(f"models--*faster-whisper-{c.model_size}*")) if cache.exists() else []
    if not hits:
        return Result(
            "Whisper weights cached",
            WARN,
            f"not in {cache} — first run downloads them (surface it in the "
            f"'loading' state, D27)",
        )
    return Result("Whisper weights cached", PASS, str(hits[0]))


def check_loopback() -> Result:
    """A WASAPI loopback device must exist; the import is lazy (D3, D22)."""
    try:
        import pyaudiowpatch as pyaudio  # noqa: PLC0415 — lazy on purpose (D22)
    except ImportError as exc:
        return Result(
            "WASAPI loopback device",
            FAIL,
            f"pyaudiowpatch unavailable ({exc.__class__.__name__}) — Windows-only "
            f"dependency, marked sys_platform == 'win32'",
            windows_only=True,
        )

    pa = None
    try:
        pa = pyaudio.PyAudio()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        loopbacks = [d for d in pa.get_loopback_device_info_generator()]
        match = next(
            (d for d in loopbacks if default_out["name"] in d["name"]),
            loopbacks[0] if loopbacks else None,
        )
        if match is None:
            return Result(
                "WASAPI loopback device",
                FAIL,
                "no loopback device found for the default output",
                windows_only=True,
            )
        return Result(
            "WASAPI loopback device",
            PASS,
            f"{match['name']} · {int(match['defaultSampleRate'])} Hz · "
            f"{int(match['maxInputChannels'])} ch (never hard-coded, D41)",
            windows_only=True,
            extra={"index": match["index"], "rate": match["defaultSampleRate"]},
        )
    except Exception as exc:  # noqa: BLE001
        return Result("WASAPI loopback device", FAIL, f"{type(exc).__name__}: {exc}", windows_only=True)
    finally:
        if pa is not None:
            pa.terminate()


def check_api_key(c: cfg.Config) -> Result:
    if not c.google_translate_api_key:
        return Result(
            "Translation API key",
            FAIL,
            "GOOGLE_TRANSLATE_API_KEY unset — copy .env.example to .env (D37)",
        )
    key = c.google_translate_api_key
    return Result("Translation API key", PASS, f"present ({len(key)} chars, ends …{key[-4:]})")


def check_translation(c: cfg.Config, allow_network: bool) -> Result:
    """One 5-character live translation (D40). Also proves the console can
    print the non-ASCII answer, which is the D41 cp1252 failure."""
    if not allow_network:
        return Result("Live translation", SKIP, "--no-net")
    if not c.google_translate_api_key:
        return Result("Live translation", SKIP, "no API key")
    probe = "¿Qué?"  # 5 characters, non-ASCII on purpose
    try:
        import httpx

        r = httpx.post(
            c.translate_url,
            params={"key": c.google_translate_api_key},
            data={"q": probe, "source": "es", "target": "en", "format": "text"},
            timeout=c.translate_timeout_s,
        )
        r.raise_for_status()
        out = r.json()["data"]["translations"][0]["translatedText"]
    except Exception as exc:  # noqa: BLE001
        return Result("Live translation", FAIL, f"{type(exc).__name__}: {exc}")
    return Result("Live translation", PASS, f"{probe} → {out}  ({len(probe)} chars billed)")


def check_writable_dirs(c: cfg.Config) -> Result:
    """The persisted character counter and the JSONL log need a home (D31, D34)."""
    try:
        c.logs_dir.mkdir(parents=True, exist_ok=True)
        probe = c.logs_dir / ".doctor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Result("State directory writable", FAIL, f"{c.data_dir}: {exc}")
    return Result("State directory writable", PASS, str(c.data_dir))


def check_config(c: cfg.Config) -> Result:
    """Echo the derived D25 values and the word age they predict."""
    predicted = c.predicted_word_age_ms(rtf=0.5) / 1000
    ok = predicted < 7.0
    return Result(
        "Latency budget (D25)",
        PASS if ok else FAIL,
        f"silence={c.silence_threshold_ms} max_utt={c.max_utterance_ms} "
        f"min={c.min_utterance_ms} floor={c.max_utterance_floor_ms} "
        f"queue={c.asr_queue_maxsize} → predicted p95 word age at RTF 0.5: "
        f"{predicted:.1f} s (target < 7 s)",
        extra={"predicted_word_age_s": round(predicted, 2)},
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_checks(c: cfg.Config, allow_network: bool = True) -> list[Result]:
    checks: list[Callable[[], Result]] = [
        check_python,
        check_utf8,
        lambda: check_lockfile(c),
        check_no_torch,
        lambda: check_config(c),
        lambda: check_silero_asset(c),
        check_cuda,
        check_cuda_libs,
        lambda: check_model_choice(c),
        lambda: check_model_cache(c),
        check_loopback,
        lambda: check_writable_dirs(c),
        lambda: check_api_key(c),
        lambda: check_translation(c, allow_network),
    ]
    return [fn() for fn in checks]


def render(results: list[Result]) -> str:
    width = max(len(r.name) for r in results)
    lines = [
        f"speech-translator doctor · {platform.system()} {platform.machine()} · "
        f"target platform: Windows (D16)",
        "",
    ]
    for r in results:
        mark = f"[{r.status}]"
        note = "  (expected here — Windows is the target)" if r.expected_failure else ""
        lines.append(f"{mark:>7}  {r.name.ljust(width)}  {r.detail}{note}")

    real_failures = [r for r in results if r.status == FAIL and not r.expected_failure]
    expected = [r for r in results if r.expected_failure]
    warns = [r for r in results if r.status == WARN]

    lines.append("")
    summary = (
        f"{sum(1 for r in results if r.status == PASS)} passed · "
        f"{len(real_failures)} failed · {len(expected)} expected-fail on this platform · "
        f"{len(warns)} warning"
    )
    lines.append(summary)
    if expected and not IS_WINDOWS:
        lines.append(
            "Linux dev box: CUDA and loopback cannot pass here by design (D17, D22). "
            "Run this again on the Windows machine after every git pull."
        )
    if real_failures:
        lines.append("Fix the failed checks above before running the pipeline.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    parser = argparse.ArgumentParser(
        prog="python -m speech_translator.doctor",
        description="Preflight check for the speech translator (D40).",
    )
    parser.add_argument("--no-net", action="store_true", help="skip the live translation call")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    c = cfg.load_config()
    results = run_checks(c, allow_network=not args.no_net)

    if args.json:
        payload = {
            "platform": sys.platform,
            "checks": [
                {
                    "name": r.name,
                    "status": r.status,
                    "detail": r.detail,
                    "expected_failure": r.expected_failure,
                    **({"extra": r.extra} if r.extra else {}),
                }
                for r in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render(results))

    return 1 if any(r.status == FAIL and not r.expected_failure for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
