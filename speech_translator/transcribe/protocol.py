"""Picklable IPC message types for the Segmenter → Transcriber queue (D15, D27).

All types are frozen dataclasses so they serialise cleanly through
multiprocessing.Queue with a spawn start method on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    """One accepted transcription, per INTERFACES.md §3."""

    utterance_id: str
    text: str
    language: str  # ISO 639-1
    language_confidence: float
    asr_ms: int
    no_speech_prob: float
    avg_logprob: float


@dataclass(frozen=True)
class WorkerReady:
    """Sent from the worker to the main process after the model loads (D27)."""


@dataclass(frozen=True)
class WorkerResult:
    """One transcription result. transcript is None when rejected by the guard (D30)."""

    transcript: Transcript | None


@dataclass(frozen=True)
class WorkerError:
    """Sent instead of WorkerReady when model load fails."""

    message: str
    detail: str  # full traceback text
