"""Windowed language identification (D32).

Locking on the first utterance alone is a coin flip when that utterance is
"okay, hi".  Accumulate confidence-weighted votes over the first ~10 s of
*speech* (not wall clock — silence must not count), then lock for the session.
"""

from __future__ import annotations

from collections import defaultdict

from speech_translator import config as cfg


class LidAccumulator:
    """Accumulates per-utterance LID votes until the speech window is full."""

    def __init__(self, config: cfg.Config | None = None) -> None:
        self._config = config or cfg.load_config()
        self._votes: dict[str, float] = defaultdict(float)  # lang → total weighted conf
        self._speech_ms: int = 0
        self._locked: str | None = None

    def update(self, language: str, confidence: float, speech_ms: int) -> None:
        """Record one LID observation.  Locks automatically at the speech window."""
        if self._locked is not None:
            return
        self._votes[language] += confidence
        self._speech_ms += speech_ms
        if self._speech_ms >= self._config.lid_window_speech_ms:
            self._locked = max(self._votes, key=self._votes.__getitem__)

    @property
    def locked_language(self) -> str | None:
        """The locked language, or None while still accumulating."""
        return self._locked

    def best_language(self) -> tuple[str, float] | None:
        """Current best-guess language and its share of total weighted confidence.

        Returns None before the first observation.  Useful for UI hints before
        locking is complete.
        """
        if not self._votes:
            return None
        lang = max(self._votes, key=self._votes.__getitem__)
        total = sum(self._votes.values())
        return lang, self._votes[lang] / total if total > 0 else 0.0

    def reset(self) -> None:
        """Clear all state for a new session."""
        self._votes.clear()
        self._speech_ms = 0
        self._locked = None
