"""Picklable IPC message types for the translation pipeline (INTERFACES.md §4–5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Sentence:
    """One caption-sized unit of speech, per INTERFACES.md §4."""

    id: str                                       # e.g. "u_0042.s1"
    utterance_id: str
    text: str
    language: str                                 # ISO 639-1
    preceding_silence_ms: int                    # non-zero only on first sentence of each utterance
    closed_by: Literal["silence", "max_length"]
    is_flush: bool                                # True when emitted by flush(), not a sentence boundary


@dataclass(frozen=True)
class Translation:
    """One translated sentence, per INTERFACES.md §5."""

    sentence_id: str
    source_text: str
    target_text: str | None                      # None on failure; UI shows source only
    source_lang: str
    target_lang: str
    mt_ms: int
    mt_error: str | None
