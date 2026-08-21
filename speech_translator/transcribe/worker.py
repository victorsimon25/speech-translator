"""Transcription worker process and its main-process manager (D15, D27).

Three layers so each can be tested without the others:

  _load_model()         deferred ctranslate2 import; Windows DLL fix (D60)
  _run_worker_loop()    core loop; accepts any model-like object for tests
  worker_main()         subprocess entry point; ties the two together
  TranscribeWorker      main-process lifecycle manager
"""

from __future__ import annotations

import json
import math
import multiprocessing
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from speech_translator import config as cfg
from speech_translator.windows_cuda import add_cuda_dll_directories

from .guard import is_accepted
from .lid import LidAccumulator
from .protocol import Transcript, WorkerError, WorkerReady, WorkerResult


def _load_model(config: cfg.Config) -> Any:
    """Load WhisperModel.  Must be called in the worker process (D60)."""
    add_cuda_dll_directories()  # PATH + os.add_dll_directory before ctranslate2
    from faster_whisper import WhisperModel  # noqa: PLC0415 — intentionally deferred
    return WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type,
        download_root=str(config.model_cache_dir),
    )


def _run_worker_loop(
    in_queue: Any,
    out_queue: Any,
    model: Any,
    lid: LidAccumulator,
    config: cfg.Config,
    log_path: Path | None,
) -> None:
    """Core transcription loop.  Accepts any queue-like and any model-like object.

    Separated from worker_main so tests can call it directly with a fake model
    and stdlib queue, without spawning a subprocess or touching CUDA.
    """
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: WPS515

    try:
        while True:
            item = in_queue.get()
            if item is None:  # shutdown sentinel
                break

            utterance = item
            audio_ms = utterance.end_ms - utterance.start_ms
            audio = (
                np.frombuffer(utterance.pcm, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            kwargs: dict[str, Any] = {
                "beam_size": config.beam_size,
                "condition_on_previous_text": config.condition_on_previous_text,
                "vad_filter": config.vad_filter,
            }
            locked = lid.locked_language
            if locked is not None:
                kwargs["language"] = locked  # pin after LID window (D32)

            t0 = time.perf_counter()
            segments_gen, info = model.transcribe(audio, **kwargs)
            # Drain the generator here. model.transcribe() is lazy: timing the
            # call without consuming the iterator times only the setup (D58).
            segments = list(segments_gen)
            asr_ms = round((time.perf_counter() - t0) * 1000)

            if segments:
                no_speech_prob = max(seg.no_speech_prob for seg in segments)
                avg_logprob = sum(seg.avg_logprob for seg in segments) / len(segments)
            else:
                no_speech_prob = 1.0
                avg_logprob = float("-inf")

            text = "".join(seg.text for seg in segments).strip()
            accepted = is_accepted(no_speech_prob, avg_logprob, config) and bool(text)

            # Update LID regardless of acceptance: a hallucinated "Thank you."
            # on a real-language utterance still carries language evidence.
            lid.update(info.language, info.language_probability, audio_ms)

            if log_file is not None:
                rtf = round(asr_ms / audio_ms, 4) if audio_ms > 0 else 0.0
                record: dict[str, Any] = {
                    "utterance_id": utterance.id,
                    "audio_ms": audio_ms,
                    "asr_ms": asr_ms,
                    "rtf": rtf,
                    "language": info.language,
                    "no_speech_prob": round(no_speech_prob, 4),
                    "avg_logprob": (
                        round(avg_logprob, 4) if math.isfinite(avg_logprob) else None
                    ),
                    "accepted": accepted,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

            if accepted:
                out_queue.put(
                    WorkerResult(
                        transcript=Transcript(
                            utterance_id=utterance.id,
                            text=text,
                            language=info.language,
                            language_confidence=info.language_probability,
                            asr_ms=asr_ms,
                            no_speech_prob=no_speech_prob,
                            avg_logprob=avg_logprob,
                        )
                    )
                )
            else:
                out_queue.put(WorkerResult(transcript=None))
    finally:
        if log_file is not None:
            log_file.close()


def worker_main(in_queue: Any, out_queue: Any, config: cfg.Config) -> None:
    """Subprocess entry point (D27).  Called by TranscribeWorker via spawn.

    Imports speech_translator (triggering force_utf8 — D42) before touching
    CUDA.  Sends WorkerReady on success or WorkerError on failure so the main
    process never hangs waiting for a model that silently OOMed.
    """
    try:
        model = _load_model(config)
        out_queue.put(WorkerReady())
    except Exception as exc:
        out_queue.put(WorkerError(message=str(exc), detail=traceback.format_exc()))
        return

    lid = LidAccumulator(config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = config.logs_dir / f"asr_{timestamp}.jsonl"
    _run_worker_loop(in_queue, out_queue, model, lid, config, log_path)


class TranscribeWorker:
    """Manages the transcription subprocess from the main asyncio process (D27).

    Usage::

        worker = TranscribeWorker()
        worker.start()                   # blocks until model loaded or error
        worker.in_queue.put(utterance)
        result = worker.out_queue.get()  # WorkerResult or WorkerError
        worker.stop()
    """

    def __init__(self, config: cfg.Config | None = None) -> None:
        self._config = config or cfg.load_config()
        self._state = "idle"
        self._last_error: WorkerError | None = None
        self._process: Any = None
        self._in_queue: Any = None
        self._out_queue: Any = None

    def start(self) -> None:
        """Spawn the worker and block until WorkerReady or WorkerError (D27).

        First run downloads Whisper weights; allow up to 120 s.
        """
        ctx = multiprocessing.get_context("spawn")
        self._in_queue = ctx.Queue(maxsize=self._config.asr_queue_maxsize)
        self._out_queue = ctx.Queue()
        self._state = "loading"
        self._process = ctx.Process(
            target=worker_main,
            args=(self._in_queue, self._out_queue, self._config),
            daemon=True,
        )
        self._process.start()
        msg = self._out_queue.get(timeout=120)
        if isinstance(msg, WorkerError):
            self._state = "error"
            self._last_error = msg
        else:
            self._state = "ready"

    def stop(self) -> None:
        """Send the shutdown sentinel and wait for clean exit."""
        if self._in_queue is not None:
            self._in_queue.put(None)
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
        self._state = "idle"

    @property
    def in_queue(self) -> Any:
        """Bounded input queue (maxsize=asr_queue_maxsize).  Put Utterances here."""
        return self._in_queue

    @property
    def out_queue(self) -> Any:
        """Output queue.  Contains WorkerResult objects."""
        return self._out_queue

    @property
    def state(self) -> str:
        """One of: idle | loading | ready | error."""
        return self._state

    @property
    def last_error(self) -> WorkerError | None:
        return self._last_error
