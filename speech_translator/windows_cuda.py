"""Make the pip-wheel CUDA DLLs findable on Windows.

D41 takes cuBLAS and cuDNN from the ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12``
wheels rather than hand-copied DLLs, because wheels are pinned by the lockfile
and hand-copied files are not reproducible. The wheels install to
``site-packages/nvidia/<component>/bin`` on Windows — and, unlike PyTorch,
nothing puts those directories on the DLL search path. Without this,
``import ctranslate2`` fails with a bare "DLL load failed", which points at
CTranslate2 rather than at the missing cuDNN.

No-op on Linux. Import is safe everywhere (D22).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ADDED: list[str] | None = None


def cuda_dll_dirs() -> list[Path]:
    """Directories inside the installed nvidia-* wheels that hold DLLs."""
    if sys.platform != "win32":
        return []
    try:
        import nvidia  # noqa: PLC0415 — only exists when the wheels are installed
    except ImportError:
        return []
    dirs: list[Path] = []
    for root in map(Path, nvidia.__path__):
        for bin_dir in sorted(root.glob("*/bin")):
            if any(bin_dir.glob("*.dll")):
                dirs.append(bin_dir)
    return dirs


def add_cuda_dll_directories() -> list[str]:
    """Register the wheel DLL directories with the loader. Idempotent.

    Call this **before** importing ``ctranslate2`` — in the doctor, and in the
    transcription worker process, which on Windows is spawned and re-imports
    the package from scratch (D27).
    """
    global _ADDED
    if _ADDED is not None:
        return _ADDED
    added: list[str] = []
    for d in cuda_dll_dirs():
        try:
            os.add_dll_directory(str(d))  # type: ignore[attr-defined]  # Windows only
            added.append(str(d))
        except (OSError, AttributeError):
            continue
    _ADDED = added
    return added
