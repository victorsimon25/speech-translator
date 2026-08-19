"""``python -m speech_translator.tools.benchmark``.

The harness that executes ``docs/BENCHMARK.md`` (D51). Seven configurations, ten
sustained minutes each, per-chunk timing, first-minute vs last-minute RTF, VRAM
polled across the run, and the results table at the end.

**It reports; it does not decide.** The model choice is a measurement made by
reading the table, and the selection rule — "the fastest that is accurate enough"
(D25/D26) — needs a listening test this file cannot perform. Nothing here writes
``model_size`` or ``compute_type`` into config.

**Written and dry-run on Linux, executed on Windows** (D51, the same split as
D48): every loop, timer, percentile and table row is platform-neutral, and the
only Windows-specific thing about it is that a GPU answers. ``--engine stub``
runs the whole harness with no weights, no network and no GPU, which is what
makes the dry run part of ``pytest`` rather than something someone remembers to
do (D58).

Three things it is deliberately strict about:

* **The CUDA guard runs before any timing.** ``get_cuda_device_count()`` must
  return 1 with ``--device cuda``, because there is no CPU fallback (D36) and a
  silent fallback would produce a whole table of fiction.
* **``MT_roundtrip`` is measured, not assumed** (D51). Ten real calls, p95. With
  no API key the term is recorded as *missing* and gate 3 reports ``unknown``;
  the guessed 300 ms is never substituted into a gate.
* **A configuration that OOMs is a failed row, not a dead run.** Losing rows 4-7
  because row 3 died is the failure this file was written to prevent.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .. import config as cfg
from ..audio.select import open_source
from ..audio.source import AudioSourceError
from ..logging_setup import force_utf8
from ..segment import Segmenter, SileroVad
from ..windows_cuda import add_cuda_dll_directories

# --------------------------------------------------------------------------
# The matrix and the gates — both come from docs/BENCHMARK.md, not from here
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """One row of the shortlist. `large-v3` at float16 is deliberately absent —
    ~3.1 GB of weights on a 4 GB card that is also drawing the desktop and the
    browser (D18)."""

    model_size: str
    compute_type: str

    @property
    def key(self) -> str:
        return f"{self.model_size}:{self.compute_type}"


#: The seven rows of `BENCHMARK.md`'s results table, in its order. Distil-Whisper
#: is excluded on purpose and the reason is D19, not an oversight: it is
#: English-only, and the product premise is a language the user does not speak.
DEFAULT_MATRIX: tuple[RunConfig, ...] = (
    RunConfig("small", "float16"),
    RunConfig("small", "int8_float16"),
    RunConfig("medium", "float16"),
    RunConfig("medium", "int8_float16"),
    RunConfig("large-v3-turbo", "float16"),
    RunConfig("large-v3-turbo", "int8_float16"),
    RunConfig("large-v3", "int8_float16"),
)

#: Gate 1 (BENCHMARK.md). Half, not 1.0 — the rest of the budget is translation,
#: the UI, the browser and thermal throttling.
GATE_RTF_MAX = 0.5
#: Gate 3 (D25/D26). Predicted here, measured end to end at Level 7.
GATE_WORD_AGE_MAX_S = 7.0
#: Gate 2 has no number in BENCHMARK.md — "leaves room for the desktop and the
#: browser" is a judgement. What the harness can do mechanically is check that
#: peak usage left this much of the card unclaimed, with the browser already open
#: as the method requires. Stated in the output so the rule is visible.
GATE_VRAM_HEADROOM_FRACTION = 0.10

#: Excluded from RTF and reported separately (BENCHMARK.md step 5).
WARMUP_CHUNKS = 1

MINUTE_S = 60.0


# --------------------------------------------------------------------------
# Chunking (D57)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One unit of audio handed to Whisper, as the app would hand it over."""

    index: int
    pcm: bytes
    duration_ms: int
    closed_by: str | None  # None for fixed-length chunking

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


def build_chunks(
    input_wav: str | Path,
    mode: str = "segmenter",
    chunk_ms: int = 4_000,
    config: cfg.Config | None = None,
) -> tuple[list[Chunk], dict]:
    """Cut the recording once, up front, and reuse it for every configuration.

    **Once** is the point. The Segmenter is deterministic over a WAV, so building
    the list here means all seven rows see byte-identical audio and an RTF
    difference between rows is a difference in the model, not in what was
    measured. It also keeps the VAD's own cost outside the timed region.

    ``mode="segmenter"`` is the default (D57): real utterances close on silence
    far more often than on max-length, so the true distribution skews shorter
    than 4 s, and shorter chunks amortise per-call overhead over less audio —
    which makes RTF *worse*. Fixed 4 s is optimistic, not neutral (D52).
    ``mode="fixed"`` keeps D52's bracket available for the selected row.

    The audio comes through ``open_source()`` — the same seam `record_loopback`
    and `dump_utterances` use (D48). No WAV is opened by hand anywhere here.
    """
    c = config or cfg.load_config()
    source = open_source(wav=input_wav, realtime=False)
    chunks: list[Chunk] = []

    try:
        if mode == "segmenter":
            segmenter = Segmenter(c, SileroVad())
            for u in segmenter.utterances(source.frames()):
                chunks.append(
                    Chunk(len(chunks), u.pcm, u.duration_ms, u.closed_by)
                )
            frames_seen = segmenter.frames_pushed
            speech_ms = segmenter.speech_frames * cfg.FRAME_MS
            discarded = segmenter.utterances_discarded
        elif mode == "fixed":
            per_chunk = max(1, chunk_ms // cfg.FRAME_MS)
            buf: list[bytes] = []
            frames_seen = 0
            for frame in source.frames():
                frames_seen += 1
                buf.append(frame)
                if len(buf) == per_chunk:
                    chunks.append(_fixed_chunk(len(chunks), buf))
                    buf = []
            if buf:
                # The tail is kept, not dropped: it is real audio, and its
                # shorter duration is recorded rather than assumed.
                chunks.append(_fixed_chunk(len(chunks), buf))
            speech_ms = None
            discarded = 0
        else:
            raise ValueError(f"unknown chunking mode: {mode!r}")
    finally:
        source.close()

    durations = [ch.duration_ms for ch in chunks]
    summary = {
        "mode": mode,
        "chunk_ms": chunk_ms if mode == "fixed" else None,
        "source_wav": str(input_wav),
        "count": len(chunks),
        "audio_s": round(sum(durations) / 1000.0, 3),
        "source_audio_s": round(frames_seen * cfg.FRAME_MS / 1000.0, 3),
        "speech_s": None if speech_ms is None else round(speech_ms / 1000.0, 3),
        "discarded": discarded,
        "duration_ms": {
            "min": min(durations) if durations else 0,
            "p50": round(percentile(durations, 50), 1) if durations else 0,
            "p95": round(percentile(durations, 95), 1) if durations else 0,
            "max": max(durations) if durations else 0,
        },
        "closed_by": {
            value: sum(1 for ch in chunks if ch.closed_by == value)
            for value in ("silence", "max_length")
        }
        if mode == "segmenter"
        else None,
    }
    return chunks, summary


def _fixed_chunk(index: int, frames: list[bytes]) -> Chunk:
    pcm = b"".join(frames)
    return Chunk(index, pcm, len(frames) * cfg.FRAME_MS, None)


# --------------------------------------------------------------------------
# Engines (D58)
# --------------------------------------------------------------------------


class Engine(Protocol):
    """What the timing loop needs, and nothing else."""

    def load(self) -> None: ...
    def transcribe(self, pcm: bytes) -> str: ...
    def unload(self) -> None: ...


def transcribe_kwargs(language: str) -> dict:
    """The configuration the app will actually use, or the number is fiction.

    BENCHMARK.md step 4. Frozen here and written verbatim into `results.json`,
    because benchmarking anything else measures software we are not shipping.
    """
    return {
        "language": language,          # locked after a short window at runtime (D32)
        "beam_size": 1,                # greedy, as the real-time path will be
        "condition_on_previous_text": False,  # no hallucination loops on chunks
        "vad_filter": False,           # our Segmenter owns VAD (D23)
    }


class FasterWhisperEngine:
    """`faster-whisper` on CTranslate2. The import is lazy and DLL-guarded."""

    def __init__(
        self,
        run_config: RunConfig,
        device: str,
        language: str,
        cache_dir: Path,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self.run_config = run_config
        self.device = device
        self.kwargs = transcribe_kwargs(language)
        self.cache_dir = cache_dir
        self._model_factory = model_factory
        self._model: object | None = None

    def load(self) -> None:
        if self._model_factory is None:
            # Must happen before the first CTranslate2 import on Windows (D41,
            # D43): the cuBLAS/cuDNN wheels unpack somewhere nothing puts on the
            # DLL search path, and the failure reads as "CTranslate2 is broken".
            add_cuda_dll_directories()
            from faster_whisper import WhisperModel  # noqa: PLC0415 — lazy on purpose

            factory: Callable[..., object] = WhisperModel
        else:
            factory = self._model_factory

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = factory(
            self.run_config.model_size,
            device=self.device,
            compute_type=self.run_config.compute_type,
            download_root=str(self.cache_dir),
        )

    def transcribe(self, pcm: bytes) -> str:
        if self._model is None:
            raise RuntimeError("engine.load() was not called")
        import numpy as np  # noqa: PLC0415 — kept beside its one use

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(audio, **self.kwargs)  # type: ignore[attr-defined]
        # `segments` is a GENERATOR and decoding does not happen until it is
        # consumed. Timing the call without draining it here times the setup and
        # nothing else — a benchmark that reports RTF 0.002 and means it.
        return "".join(segment.text for segment in segments)

    def unload(self) -> None:
        self._model = None


class StubEngine:
    """A fake transcriber with a stated RTF. No weights, no network, no GPU.

    This is what makes the dry run a test rather than a ritual (D58): it
    exercises the chunk loop, the warm-up exclusion, the wall-clock minute split,
    the percentiles, the incremental writes, the OOM path and the table — every
    part of the harness except the part where a GPU answers.
    """

    def __init__(
        self,
        run_config: RunConfig,
        rtf: float = 0.25,
        load_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        fail_at_chunk: int | None = None,
        fail_at_load: bool = False,
        error: BaseException | None = None,
        text: str = "texto de prueba. ",
    ) -> None:
        self.run_config = run_config
        self.rtf = rtf
        self.load_s = load_s
        self._sleep = sleep
        self.fail_at_chunk = fail_at_chunk
        self.fail_at_load = fail_at_load
        self.error = error or RuntimeError(
            "CUDA failed with error out of memory"
        )
        self.text = text
        self.calls = 0
        self.loaded = False

    def load(self) -> None:
        if self.fail_at_load:
            raise self.error
        self._sleep(self.load_s)
        self.loaded = True

    def transcribe(self, pcm: bytes) -> str:
        if self.fail_at_chunk is not None and self.calls >= self.fail_at_chunk:
            raise self.error
        self.calls += 1
        audio_s = len(pcm) / (cfg.SAMPLE_RATE * cfg.SAMPLE_WIDTH_BYTES)
        self._sleep(audio_s * self.rtf)
        return self.text

    def unload(self) -> None:
        self.loaded = False


# --------------------------------------------------------------------------
# The CUDA guard (D36)
# --------------------------------------------------------------------------


class BenchmarkAborted(RuntimeError):
    """Raised before any timing when the run could only produce fiction."""


def cuda_device_count() -> int:
    """`ctranslate2.get_cuda_device_count()`, with the DLL path set up first."""
    add_cuda_dll_directories()  # before the import, on Windows (D41, D43)
    import ctranslate2  # noqa: PLC0415 — lazy so the module imports without CUDA

    return int(ctranslate2.get_cuda_device_count())


def require_cuda(device: str, count_fn: Callable[[], int] = cuda_device_count) -> int:
    """Refuse to time anything until a GPU has answered.

    BENCHMARK.md's step 1 and PLAN.md's first acceptance criterion. There is no
    CPU fallback in this project (D36), so a silent fallback would not be slow —
    it would be seven rows of numbers describing a machine we are not shipping
    on.
    """
    if device != "cuda":
        return 0
    try:
        count = count_fn()
    except Exception as exc:  # noqa: BLE001 — the diagnosis is the point
        raise BenchmarkAborted(
            f"could not ask CTranslate2 for a CUDA device count: "
            f"{type(exc).__name__}: {exc}. On Windows this is usually the "
            f"cuBLAS/cuDNN DLL trap — run `python -m speech_translator.doctor` "
            f"(D41, D43)."
        ) from exc
    if count != 1:
        raise BenchmarkAborted(
            f"ctranslate2.get_cuda_device_count() == {count}, expected 1. "
            f"No timing has been taken. There is no CPU fallback in this "
            f"project (D36), so a run that quietly used the CPU would report "
            f"seven rows of fiction. Run `python -m speech_translator.doctor`; "
            f"pass --device cpu only for a deliberate dry run, which stamps "
            f"every row NOT A GPU MEASUREMENT."
        )
    return count


# --------------------------------------------------------------------------
# The measured MT round trip (D51)
# --------------------------------------------------------------------------

#: Short on purpose: this is billed against the 500 000-character free tier
#: (D31), and what is being timed is the round trip, not the translation.
_MT_PROBE = "¿Me escuchan bien?"


def measure_mt_roundtrip(
    config: cfg.Config,
    samples: int = 10,
    post: Callable[..., object] | None = None,
) -> dict:
    """Time ten real calls and take the p95 (D51).

    The 300 ms in D25's arithmetic was a placeholder for a live network call.
    Substituting it into a gate is exactly what D25 exists to stop, so with no
    key this returns ``p95_ms=None`` and ``source="unmeasured"`` — and gate 3
    then reports *unknown* rather than a number that looks measured.
    """
    if not config.google_translate_api_key:
        return {
            "p95_ms": None,
            "source": "unmeasured",
            "n": 0,
            "samples_ms": [],
            "chars_billed": 0,
            "error": "GOOGLE_TRANSLATE_API_KEY unset — the MT term of the word-age "
                     "formula is missing, not 300 ms (D51)",
        }

    if post is None:
        import httpx  # noqa: PLC0415

        post = httpx.post

    timings: list[float] = []
    error: str | None = None
    for _ in range(samples):
        t0 = time.perf_counter()
        try:
            response = post(
                config.translate_url,
                params={"key": config.google_translate_api_key},
                data={
                    "q": _MT_PROBE,
                    "source": "es",
                    "target": config.target_lang,
                    "format": "text",
                },
                timeout=config.translate_timeout_s,
            )
            response.raise_for_status()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            break
        timings.append((time.perf_counter() - t0) * 1000.0)

    if not timings:
        return {
            "p95_ms": None,
            "source": "unmeasured",
            "n": 0,
            "samples_ms": [],
            "chars_billed": 0,
            "error": error or "no samples",
        }

    return {
        "p95_ms": round(percentile(timings, 95), 1),
        "source": "measured",
        "n": len(timings),
        "samples_ms": [round(t, 1) for t in timings],
        "chars_billed": len(_MT_PROBE) * len(timings),
        "error": error,
    }


# --------------------------------------------------------------------------
# GPU polling
# --------------------------------------------------------------------------

_SMI_GPU_FIELDS = ("memory.used", "memory.total", "temperature.gpu", "clocks.sm", "power.draw")


class GpuPoller:
    """`nvidia-smi` at 1 Hz across the run, exactly as BENCHMARK.md prescribes.

    A subprocess rather than a new dependency: `pynvml` would mean touching the
    lockfile (D39) for something the method already specifies as a shell command.
    No `nvidia-smi` on the machine — the Linux dev box — means `available` is
    False and the VRAM columns come out null, which is the honest answer and not
    a crash.
    """

    def __init__(
        self,
        interval_s: float = 1.0,
        runner: Callable[[list[str]], str | None] | None = None,
    ) -> None:
        self.interval_s = interval_s
        self._run = runner or _run_smi
        self.available = shutil.which("nvidia-smi") is not None if runner is None else True
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._tag = ""
        self._lock = threading.Lock()

    # -- one-shot queries --------------------------------------------------

    def sample(self) -> dict | None:
        if not self.available:
            return None
        out = self._run(
            ["nvidia-smi", f"--query-gpu={','.join(_SMI_GPU_FIELDS)}",
             "--format=csv,noheader,nounits"]
        )
        if not out:
            return None
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        if len(parts) < len(_SMI_GPU_FIELDS):
            return None
        record = {
            "t": round(time.time(), 3),
            "memory_used_mb": _to_float(parts[0]),
            "memory_total_mb": _to_float(parts[1]),
            "temperature_c": _to_float(parts[2]),
            "sm_clock_mhz": _to_float(parts[3]),
            "power_w": _to_float(parts[4]),
            "own_memory_mb": self._own_memory_mb(),
        }
        return record

    def _own_memory_mb(self) -> float | None:
        """Our process's slice, so "peak VRAM" can be split from the browser's."""
        out = self._run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"]
        )
        if not out:
            return None
        pid = os.getpid()
        for line in out.strip().splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) >= 2 and fields[0].isdigit() and int(fields[0]) == pid:
                return _to_float(fields[1])
        return None

    def describe(self) -> dict:
        """Half the hardware survey, collected rather than estimated."""
        if not self.available:
            return {"available": False}
        out = self._run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"]
        )
        info: dict = {"available": True}
        if out:
            parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                info.update(
                    name=parts[0],
                    driver_version=parts[1],
                    memory_total_mb=_to_float(parts[2]),
                )
        return info

    # -- the background poll ----------------------------------------------

    def start(self, jsonl_path: Path | None = None, tag: str = "") -> None:
        if not self.available:
            return
        self._stop.clear()
        self.samples = []
        if jsonl_path is not None:
            self._file = jsonl_path.open("a", encoding="utf-8")
        self._tag = tag
        self._thread = threading.Thread(target=self._loop, name="gpu-poller", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            record = self.sample()
            if record is not None:
                record["tag"] = self._tag
                with self._lock:
                    self.samples.append(record)
                    if self._file is not None:
                        self._file.write(json.dumps(record) + "\n")
                        self._file.flush()
            self._stop.wait(self.interval_s)

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None
        with self._lock:
            return list(self.samples)


def _run_smi(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Arithmetic — small, exact, and tested rather than trusted
# --------------------------------------------------------------------------


def percentile(values: Iterable[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method).

    Written out rather than imported so the p95 in a gate has a definition that
    can be read and hand-checked.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def rtf_of(records: list[dict]) -> float | None:
    """`sum(processing) / sum(audio)` — over the whole run, not a slice (step 6)."""
    audio = sum(r["audio_s"] for r in records)
    if audio <= 0:
        return None
    return sum(r["proc_s"] for r in records) / audio


def window(records: list[dict], start_s: float, end_s: float) -> list[dict]:
    """Chunks whose processing *started* inside a wall-clock window.

    Start-based membership, so a chunk belongs to exactly one window and no
    chunk is counted twice at a boundary.
    """
    return [r for r in records if start_s <= r["wall_start_s"] < end_s]


def word_age_ms(silence_ms: int, mt_ms: float, duration_ms: float, rtf: float) -> float:
    """D25's budget, evaluated: `silence + MT + D x (1 + RTF)`.

    A **prediction**. Level 7 measures it end to end off the JSONL log, where
    queue wait and the browser competing for the GPU are visible (D51). Every
    caller of this function labels its output *predicted*.
    """
    return silence_ms + mt_ms + duration_ms * (1.0 + rtf)


_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda_error_out_of_memory",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "failed to allocate",
    "bad_alloc",
)


def classify_failure(exc: BaseException) -> str:
    """`oom` or `error`. The realistic failure on CUDA is memory, not slowness."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "oom" if any(marker in text for marker in _OOM_MARKERS) else "error"


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@dataclass
class Harness:
    """Everything the timing loop needs, injectable so tests need no GPU."""

    chunks: list[Chunk]
    minutes: float = 10.0
    device: str = "cuda"
    language: str = "es"
    config: cfg.Config = field(default_factory=cfg.load_config)
    out_dir: Path = Path(".")
    engine_factory: Callable[[RunConfig], Engine] | None = None
    poller: GpuPoller | None = None
    mt: dict = field(default_factory=lambda: {"p95_ms": None, "source": "unmeasured"})
    clock: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep
    settle_s: float = 3.0
    on_event: Callable[[str], None] | None = None

    def _say(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)

    # -- one configuration -------------------------------------------------

    def run_one(self, run_config: RunConfig) -> dict:
        """Ten sustained minutes of one configuration. Never raises for a
        model-side failure: the row is the report, including when it is a
        failure (D51)."""
        row: dict = {
            "model_size": run_config.model_size,
            "compute_type": run_config.compute_type,
            "device": self.device,
            "not_a_gpu_measurement": self.device != "cuda",
            "status": "ok",
            "failure": None,
            "error": None,
        }
        records: list[dict] = []
        gpu_samples: list[dict] = []

        idle = self.poller.sample() if self.poller is not None else None
        row["idle_vram_mb"] = idle.get("memory_used_mb") if idle else None
        row["total_vram_mb"] = idle.get("memory_total_mb") if idle else None

        # The metrics are computed *after* the teardown below, not inside it:
        # the GPU poller's samples do not exist until it has been stopped, and a
        # row summarised before then reports no VRAM at all.
        gpu_samples = self._measure(run_config, row, records)
        row.update(self._gpu_metrics(gpu_samples))
        row.update(self._timing_metrics(records))
        row.update(self._gate_metrics(row, records))
        return row

    def _measure(self, run_config: RunConfig, row: dict, records: list[dict]) -> list[dict]:
        """Load, loop, tear down. Fills `row` and `records`; returns GPU samples."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = self.out_dir / f"chunks-{_slug(run_config.key)}.jsonl"
        gpu_path = self.out_dir / "gpu.jsonl"
        gpu_samples: list[dict] = []

        factory = self.engine_factory or self._default_engine
        engine = factory(run_config)

        if self.poller is not None:
            self.poller.start(gpu_path, tag=run_config.key)

        chunk_file = chunk_path.open("w", encoding="utf-8")
        try:
            # Model load is timed and then EXCLUDED from RTF (step 5). It is a
            # one-off startup cost the app pays once per session, and folding it
            # into a throughput number would flatter the small models.
            t_load = self.clock()
            try:
                engine.load()
            except Exception as exc:  # noqa: BLE001 — a failed row, not a dead run
                row.update(
                    status="failed",
                    failure=classify_failure(exc),
                    error=f"{type(exc).__name__}: {exc}",
                    stage="load",
                    load_s=round(self.clock() - t_load, 3),
                )
                self._say(f"  {run_config.key}: load failed ({row['failure']}) — {row['error']}")
                return gpu_samples
            row["load_s"] = round(self.clock() - t_load, 3)

            deadline_s = self.minutes * MINUTE_S
            t0 = self.clock()
            index = 0
            samples: list[dict] = []
            try:
                while self.clock() - t0 < deadline_s:
                    chunk = self.chunks[index % len(self.chunks)]
                    lap = index // len(self.chunks)
                    start = self.clock() - t0
                    text = engine.transcribe(chunk.pcm)
                    end = self.clock() - t0
                    record = {
                        "config": run_config.key,
                        "index": index,
                        "lap": lap,
                        "chunk": chunk.index,
                        "audio_s": chunk.duration_s,
                        "wall_start_s": round(start, 6),
                        "wall_end_s": round(end, 6),
                        "proc_s": round(end - start, 6),
                        "rtf": round((end - start) / chunk.duration_s, 6)
                        if chunk.duration_s
                        else None,
                        "warmup": index < WARMUP_CHUNKS,
                        "text_chars": len(text),
                    }
                    records.append(record)
                    chunk_file.write(json.dumps(record) + "\n")
                    chunk_file.flush()
                    if len(samples) < 5 and lap == 0:
                        # Kept so the subjective accuracy note BENCHMARK.md asks
                        # for has material to judge, in the demo's own language.
                        samples.append(
                            {"chunk": chunk.index, "audio_ms": chunk.duration_ms, "text": text}
                        )
                    index += 1
            except Exception as exc:  # noqa: BLE001
                row.update(
                    status="failed",
                    failure=classify_failure(exc),
                    error=f"{type(exc).__name__}: {exc}",
                    stage="transcribe",
                )
                self._say(
                    f"  {run_config.key}: failed after {len(records)} chunks "
                    f"({row['failure']}) — {row['error']}"
                )
            row["samples"] = samples
            return gpu_samples
        finally:
            chunk_file.close()
            engine.unload()
            # Drop the CUDA context before the next row loads a model, and give
            # the driver a moment to hand the memory back — a leaked context
            # poisons every row after it.
            gc.collect()
            self.sleep(self.settle_s)
            if self.poller is not None:
                # `extend`, not rebind: the return value above is this list, and
                # rebinding here would hand the caller the empty one.
                gpu_samples.extend(self.poller.stop())
                after = self.poller.sample()
                row["vram_after_unload_mb"] = after.get("memory_used_mb") if after else None

    def _default_engine(self, run_config: RunConfig) -> Engine:
        return FasterWhisperEngine(
            run_config,
            device=self.device,
            language=self.language,
            cache_dir=self.config.model_cache_dir,
        )

    # -- the numbers -------------------------------------------------------

    def _timing_metrics(self, records: list[dict]) -> dict:
        warm = [r for r in records if r["warmup"]]
        timed = [r for r in records if not r["warmup"]]
        out: dict = {
            "chunks_processed": len(records),
            "warmup_s": round(warm[0]["proc_s"], 3) if warm else None,
            "warmup_rtf": round(warm[0]["rtf"], 3) if warm and warm[0]["rtf"] else None,
            "wall_clock_s": round(records[-1]["wall_end_s"], 3) if records else 0.0,
            "audio_s": round(sum(r["audio_s"] for r in timed), 3),
            # Chunks processed over chunks available: 60 calls against a 4-chunk
            # recording is 15 laps, not 14.75 — the last chunk is finished.
            "laps": round(len(records) / max(1, len(self.chunks)), 2),
            "rtf": None,
            "rtf_first_min": None,
            "rtf_last_min": None,
            "rtf_p95_chunk": None,
            "throttling_penalty": None,
            "minute_windows_overlap": False,
        }
        if not timed:
            return out

        out["rtf"] = _round(rtf_of(timed), 4)
        out["rtf_p95_chunk"] = _round(percentile([r["rtf"] for r in timed], 95), 4)

        total = records[-1]["wall_end_s"]
        # The warm-up chunk lands inside the first minute and would dominate it,
        # which is the opposite of what a throttling comparison wants — so the
        # first-minute figure is taken over `timed`, not over `records`.
        first = window(timed, 0.0, MINUTE_S)
        last = window(timed, max(0.0, total - MINUTE_S), total + 1.0)
        out["rtf_first_min"] = _round(rtf_of(first), 4)
        out["rtf_last_min"] = _round(rtf_of(last), 4)
        out["minute_windows_overlap"] = total < 2 * MINUTE_S
        if out["rtf_first_min"] and out["rtf_last_min"]:
            out["throttling_penalty"] = round(
                out["rtf_last_min"] / out["rtf_first_min"] - 1.0, 4
            )
        return out

    def _gpu_metrics(self, samples: list[dict]) -> dict:
        if not samples:
            return {
                "peak_vram_mb": None,
                "peak_own_vram_mb": None,
                "peak_temp_c": None,
                "min_sm_clock_mhz": None,
                "peak_power_w": None,
                "gpu_samples": 0,
            }
        used = [s["memory_used_mb"] for s in samples if s.get("memory_used_mb") is not None]
        own = [s["own_memory_mb"] for s in samples if s.get("own_memory_mb") is not None]
        temps = [s["temperature_c"] for s in samples if s.get("temperature_c") is not None]
        clocks = [s["sm_clock_mhz"] for s in samples if s.get("sm_clock_mhz") is not None]
        power = [s["power_w"] for s in samples if s.get("power_w") is not None]
        return {
            "peak_vram_mb": max(used) if used else None,
            "peak_own_vram_mb": max(own) if own else None,
            "peak_temp_c": max(temps) if temps else None,
            # The throttling story in one number besides RTF: where the clock
            # bottomed out over ten minutes.
            "min_sm_clock_mhz": min(clocks) if clocks else None,
            "peak_power_w": max(power) if power else None,
            "gpu_samples": len(samples),
        }

    def _gate_metrics(self, row: dict, records: list[dict]) -> dict:
        """Fill the gate columns mechanically. Selecting a row is not done here."""
        mt_ms = self.mt.get("p95_ms")
        c = self.config
        out: dict = {
            "mt_p95_ms": mt_ms,
            "mt_source": self.mt.get("source", "unmeasured"),
            "word_age_p95_ms": None,
            "word_age_at_cap_ms": None,
            "gate_rtf": None,
            "gate_vram": None,
            "gate_word_age": None,
        }

        rtf = row.get("rtf")
        last = row.get("rtf_last_min")
        if row["status"] == "failed":
            out["gate_rtf"] = False
        elif rtf is not None:
            # Sustained: the whole-run figure *and* the last minute, because a
            # run that only fails the gate after it heats up still fails it.
            out["gate_rtf"] = rtf < GATE_RTF_MAX and (last is None or last < GATE_RTF_MAX)

        peak = row.get("peak_vram_mb")
        total = row.get("total_vram_mb")
        if row["status"] == "failed" and row.get("failure") == "oom":
            out["gate_vram"] = False
        elif peak is not None and total:
            out["gate_vram"] = peak <= total * (1.0 - GATE_VRAM_HEADROOM_FRACTION)
            out["vram_headroom_mb"] = round(total - peak, 1)

        # Gate 3, PREDICTED — never reported as a measurement (D51).
        if mt_ms is not None and rtf is not None and row["status"] != "failed":
            per_chunk = self._word_ages(mt_ms, records)
            if per_chunk:
                out["word_age_p95_ms"] = round(percentile(per_chunk, 95), 1)
            out["word_age_at_cap_ms"] = round(
                word_age_ms(
                    c.silence_threshold_ms,
                    mt_ms,
                    c.max_utterance_ms,
                    row.get("rtf_p95_chunk") or rtf,
                ),
                1,
            )
            if out["word_age_p95_ms"] is not None:
                out["gate_word_age"] = out["word_age_p95_ms"] < GATE_WORD_AGE_MAX_S * 1000

        return out

    def _word_ages(self, mt_ms: float, records: list[dict]) -> list[float]:
        """Word age per chunk, from that chunk's own D and its own RTF.

        A real percentile over the distribution the app will actually produce,
        rather than one point estimate at the cap. The cap bound is reported
        beside it as `word_age_at_cap_ms`, because that is the arithmetic D25
        used to derive `max_utterance_ms` in the first place.
        """
        return [
            word_age_ms(
                self.config.silence_threshold_ms,
                mt_ms,
                record["audio_s"] * 1000.0,
                record["rtf"],
            )
            for record in records
            if not record["warmup"] and record.get("rtf") is not None
        ]

    # -- the matrix --------------------------------------------------------

    def run_matrix(self, matrix: Iterable[RunConfig], results: dict) -> dict:
        """Run every configuration, writing after each one.

        Incremental on purpose (D51): a kill at minute 55 keeps minutes 0-54,
        and a row that OOMs is written as a failed row and the matrix carries on
        to the next. Losing rows 4-7 because row 3 died is the whole reason this
        file exists.
        """
        results.setdefault("rows", [])
        for run_config in matrix:
            self._say(f"[{run_config.key}] {self.minutes:g} min ...")
            row = self.run_one(run_config)
            results["rows"].append(row)
            write_results(self.out_dir, results)
            self._say(f"  {_row_line(row)}")
        results["finished_utc"] = _utc_now()
        write_results(self.out_dir, results)
        return results


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_results(out_dir: Path, results: dict) -> Path:
    """Rewrite `results.json` and re-render `results.md`. Called after every row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")
    return path


def _fmt(value: float | None, digits: int = 3, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def _gate(value: bool | None) -> str:
    return {True: "yes", False: "**no**", None: "unknown"}[value]


def _word_age_cell(row: dict) -> str:
    if row.get("mt_source") == "unmeasured":
        return "MT unmeasured"
    value = row.get("word_age_p95_ms")
    if value is None:
        return "—"
    marker = " *(MT assumed)*" if row.get("mt_source") == "assumed" else ""
    return f"{value / 1000:.1f} s{marker}"


def _row_line(row: dict) -> str:
    if row["status"] == "failed":
        return f"{row['model_size']}/{row['compute_type']}: FAILED ({row['failure']})"
    return (
        f"{row['model_size']}/{row['compute_type']}: "
        f"RTF {_fmt(row.get('rtf'))} "
        f"(first {_fmt(row.get('rtf_first_min'))} / last {_fmt(row.get('rtf_last_min'))}) · "
        f"peak VRAM {_fmt(row.get('peak_vram_mb'), 0, ' MB')} · "
        f"word age (predicted) {_word_age_cell(row)}"
    )


MARKDOWN_HEADER = (
    "| Model | compute_type | RTF (first min) | RTF (last min) | Peak VRAM | "
    "p95 word age *(predicted)* | Under 0.5 sustained? | Word age < 7 s? | "
    "Fits alongside browser? | Notes |"
)


def render_markdown(results: dict) -> str:
    """`BENCHMARK.md`'s table, in its columns, ready to transcribe.

    Emitted as a file so the table is **transcribed, not retyped**. The word-age
    column carries the word *predicted* in its header, here and everywhere else
    the harness prints it: a prediction reported as a measurement is the failure
    mode D51 named (D51, BENCHMARK.md gate 3).
    """
    harness = results.get("harness", {})
    lines = [
        "# Benchmark results",
        "",
        f"_Generated by `tools/benchmark.py` · {results.get('started_utc', '?')}_",
        "",
        f"- host: {results.get('host', {}).get('platform', '?')} · "
        f"python {results.get('host', {}).get('python', '?')}",
        f"- chunking: **{results.get('chunks', {}).get('mode', '?')}** · "
        f"{results.get('chunks', {}).get('count', 0)} chunks · "
        f"{results.get('chunks', {}).get('audio_s', 0):.1f} s of audio · "
        f"p50 {results.get('chunks', {}).get('duration_ms', {}).get('p50', 0):.0f} ms / "
        f"p95 {results.get('chunks', {}).get('duration_ms', {}).get('p95', 0):.0f} ms",
        f"- sustained: {harness.get('minutes', '?')} min of **wall clock** per "
        f"configuration, audio looped (D59)",
        f"- device: `{harness.get('device', '?')}` · engine: "
        f"`{harness.get('engine', '?')}` · language pinned: "
        f"`{harness.get('language', '?')}`",
        f"- MT round trip: {_mt_line(results.get('mt', {}))}",
        "",
    ]
    if harness.get("device") != "cuda" or harness.get("engine") != "faster-whisper":
        lines += [
            "> **NOT A GPU MEASUREMENT.** This run did not use CUDA with "
            "faster-whisper, so every number below describes the harness, not "
            "the demo machine (D51).",
            "",
        ]

    lines += [MARKDOWN_HEADER, "|" + "---|" * 10]
    for row in results.get("rows", []):
        if row["status"] == "failed":
            note = f"**FAILED ({row['failure']})** — {row.get('error', '')}"
            lines.append(
                f"| {row['model_size']} | {row['compute_type']} | — | — | "
                f"{_fmt(row.get('peak_vram_mb'), 0)} | — | **no** | unknown | "
                f"{_gate(row.get('gate_vram'))} | {note} |"
            )
            continue
        notes = []
        if row.get("throttling_penalty") is not None:
            notes.append(f"throttling {row['throttling_penalty']:+.1%}")
        if row.get("warmup_s") is not None:
            notes.append(f"warm-up {row['warmup_s']:.2f} s")
        if row.get("load_s") is not None:
            notes.append(f"load {row['load_s']:.1f} s (excluded)")
        if row.get("minute_windows_overlap"):
            notes.append("**run < 2 min: minute windows overlap**")
        if row.get("not_a_gpu_measurement"):
            notes.append("**NOT A GPU MEASUREMENT**")
        lines.append(
            f"| {row['model_size']} | {row['compute_type']} | "
            f"{_fmt(row.get('rtf_first_min'))} | {_fmt(row.get('rtf_last_min'))} | "
            f"{_fmt(row.get('peak_vram_mb'), 0, ' MB')} | {_word_age_cell(row)} | "
            f"{_gate(row.get('gate_rtf'))} | {_gate(row.get('gate_word_age'))} | "
            f"{_gate(row.get('gate_vram'))} | {'; '.join(notes)} |"
        )

    lines += [
        "",
        "Whole-run RTF (step 6), and the cap bound beside the distribution p95:",
        "",
        "| Model | compute_type | RTF (whole run) | RTF p95 (per chunk) | "
        "word age p95 *(predicted)* | word age at cap D=4000 ms *(predicted)* | "
        "peak own VRAM | min SM clock | peak temp |",
        "|" + "---|" * 9,
    ]
    for row in results.get("rows", []):
        lines.append(
            f"| {row['model_size']} | {row['compute_type']} | {_fmt(row.get('rtf'))} | "
            f"{_fmt(row.get('rtf_p95_chunk'))} | {_word_age_cell(row)} | "
            f"{_fmt((row.get('word_age_at_cap_ms') or 0) / 1000 or None, 1, ' s')} | "
            f"{_fmt(row.get('peak_own_vram_mb'), 0, ' MB')} | "
            f"{_fmt(row.get('min_sm_clock_mhz'), 0, ' MHz')} | "
            f"{_fmt(row.get('peak_temp_c'), 0, ' °C')} |"
        )

    lines += [
        "",
        "## What this table does not do",
        "",
        "It does not select a model. The rule is *the fastest that is accurate "
        "enough* (D25/D26), and \"enough\" is judged by listening — see the "
        "`samples` field in `results.json` for transcripts in the source "
        "language. Fill in the subjective accuracy note before choosing.",
        "",
        "The word-age column is **predicted**: measured RTF and a measured MT "
        "round trip put through `word_age = silence + MT + D x (1 + RTF)`. "
        "Level 7 measures it end to end off the JSONL log (D51).",
        "",
    ]
    return "\n".join(lines)


def _mt_line(mt: dict) -> str:
    if mt.get("source") == "measured":
        return (
            f"**measured** p95 {mt['p95_ms']:.0f} ms over {mt['n']} live calls "
            f"({mt.get('chars_billed', 0)} characters billed)"
        )
    if mt.get("source") == "assumed":
        return f"**assumed {mt['p95_ms']:.0f} ms** — not measured; gate 3 is not evidence"
    return f"**unmeasured** — {mt.get('error', 'no API key')}; gate 3 reports *unknown*"


def render(results: dict) -> str:
    """The console summary."""
    rows = results.get("rows", [])
    lines = [
        "",
        f"{len(rows)} configuration(s) · chunking "
        f"{results.get('chunks', {}).get('mode')} · "
        f"{results.get('chunks', {}).get('count', 0)} chunks · "
        f"{results.get('harness', {}).get('minutes')} min wall clock each",
        f"MT round trip: {_mt_line(results.get('mt', {}))}",
        "",
    ]
    for row in rows:
        lines.append("  " + _row_line(row))
    lines += [
        "",
        f"results: {results.get('out_dir', '?')}/results.json",
        f"table:   {results.get('out_dir', '?')}/results.md  (transcribe it into "
        f"docs/BENCHMARK.md)",
        "",
        "This harness reports; it does not choose. The rule is the fastest that "
        "is accurate enough (D25/D26) and 'enough' is a listening test — read "
        "the samples in results.json before selecting, then write model_size and "
        "compute_type into config yourself.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_matrix(text: str) -> list[RunConfig]:
    out: list[RunConfig] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"expected model:compute_type, got {item!r}")
        model, compute = item.split(":", 1)
        out.append(RunConfig(model.strip(), compute.strip()))
    if not out:
        raise ValueError("empty configuration list")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m speech_translator.tools.benchmark",
        description=(
            "Execute docs/BENCHMARK.md: 7 configurations x 10 sustained minutes, "
            "per-chunk timing, first/last-minute RTF, VRAM polling, and the "
            "results table. Reports; does not choose."
        ),
    )
    parser.add_argument("--input-wav", type=str, default=None,
                        help="the record_loopback capture to benchmark against")
    parser.add_argument("--configs", type=str, default=None,
                        help="model:compute_type,... (default: BENCHMARK.md's seven)")
    parser.add_argument("--minutes", type=float, default=10.0,
                        help="wall-clock minutes per configuration, audio looped (D59)")
    parser.add_argument("--chunking", choices=("segmenter", "fixed"), default="segmenter",
                        help="segmenter = the real utterance distribution (D57)")
    parser.add_argument("--chunk-ms", type=int, default=4000,
                        help="with --chunking fixed; 1500 is D52's short bracket")
    parser.add_argument("--language", type=str, default="es",
                        help="pinned, as the app locks it after detection (D32)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--engine", choices=("faster-whisper", "stub"), default="faster-whisper")
    parser.add_argument("--stub-rtf", type=float, default=0.25,
                        help="the RTF the stub engine pretends to have")
    parser.add_argument("--mt-samples", type=int, default=10,
                        help="live translation calls to time for the word-age term (D51)")
    parser.add_argument("--no-mt", action="store_true",
                        help="skip the MT probe; gate 3 then reports unknown")
    parser.add_argument("--assume-mt-ms", type=float, default=None,
                        help="use this MT round trip instead of measuring one. Stamped "
                             "'assumed' everywhere it appears — it is not evidence")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: var/benchmark/<timestamp>)")
    parser.add_argument("--render", type=Path, default=None,
                        help="re-render the table from a results.json and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="stub engine, cpu, 1 minute — exercises the harness, "
                             "measures nothing (D58)")
    return parser


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    args = build_parser().parse_args(argv)

    if args.render is not None:
        results = json.loads(args.render.read_text(encoding="utf-8"))
        print(render_markdown(results))
        return 0

    if args.dry_run:
        args.engine = "stub"
        args.device = "cpu"
        args.minutes = min(args.minutes, 1.0)

    if not args.input_wav:
        print(
            "--input-wav is required. BENCHMARK.md step 2 sources it from\n"
            "  python -m speech_translator.tools.record_loopback -t 600\n"
            "on the Windows machine, so the benchmark measures the real capture "
            "path.",
            file=sys.stderr,
        )
        return 2

    c = cfg.load_config()

    try:
        require_cuda(args.device)
    except BenchmarkAborted as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 3

    try:
        matrix = parse_matrix(args.configs) if args.configs else list(DEFAULT_MATRIX)
    except ValueError as exc:
        print(f"bad --configs: {exc}", file=sys.stderr)
        return 2

    if args.assume_mt_ms is not None:
        mt = {
            "p95_ms": args.assume_mt_ms,
            "source": "assumed",
            "n": 0,
            "samples_ms": [],
            "error": "assumed via --assume-mt-ms; D51 asks for ten measured calls",
        }
    elif args.no_mt:
        mt = {"p95_ms": None, "source": "unmeasured", "n": 0, "samples_ms": [],
              "error": "--no-mt"}
    else:
        print("timing the MT round trip (ten live calls, p95 — D51) ...", file=sys.stderr)
        mt = measure_mt_roundtrip(c, samples=args.mt_samples)
    print(f"MT: {_mt_line(mt)}", file=sys.stderr)

    try:
        chunks, chunk_summary = build_chunks(
            args.input_wav, mode=args.chunking, chunk_ms=args.chunk_ms, config=c
        )
    except AudioSourceError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    if not chunks:
        print(
            f"no chunks from {args.input_wav}: the Segmenter found no speech. "
            f"Check the recording with `python -m speech_translator.tools.dump_utterances "
            f"--input-wav {args.input_wav}` before spending 70 minutes on it.",
            file=sys.stderr,
        )
        return 2
    print(
        f"chunks: {chunk_summary['count']} · {chunk_summary['audio_s']:.1f} s of "
        f"{chunk_summary['source_audio_s']:.1f} s · p50 "
        f"{chunk_summary['duration_ms']['p50']:.0f} ms / p95 "
        f"{chunk_summary['duration_ms']['p95']:.0f} ms",
        file=sys.stderr,
    )

    out_dir = args.out or (c.data_dir / "benchmark" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    poller = GpuPoller()
    engine_factory: Callable[[RunConfig], Engine] | None = None
    if args.engine == "stub":
        engine_factory = lambda rc: StubEngine(rc, rtf=args.stub_rtf)  # noqa: E731

    harness = Harness(
        chunks=chunks,
        minutes=args.minutes,
        device=args.device,
        language=args.language,
        config=c,
        out_dir=out_dir,
        engine_factory=engine_factory,
        poller=poller,
        mt=mt,
        on_event=lambda message: print(message, file=sys.stderr, flush=True),
    )

    results = {
        "schema": 1,
        "started_utc": _utc_now(),
        "out_dir": str(out_dir),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "gpu": poller.describe(),
        "harness": {
            "minutes": args.minutes,
            "sustained": "wall clock, audio looped (D59)",
            "device": args.device,
            "engine": args.engine,
            "language": args.language,
            "warmup_chunks": WARMUP_CHUNKS,
            "model_load_excluded_from_rtf": True,
        },
        "transcribe_kwargs": transcribe_kwargs(args.language),
        "gates": {
            "rtf_max": GATE_RTF_MAX,
            "word_age_max_s": GATE_WORD_AGE_MAX_S,
            "word_age_is": "predicted, not measured (D51)",
            "vram_headroom_fraction": GATE_VRAM_HEADROOM_FRACTION,
        },
        "config": {
            "silence_threshold_ms": c.silence_threshold_ms,
            "max_utterance_ms": c.max_utterance_ms,
            "min_utterance_ms": c.min_utterance_ms,
        },
        "mt": mt,
        "chunks": chunk_summary,
        "rows": [],
    }
    write_results(out_dir, results)

    harness.run_matrix(matrix, results)
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
