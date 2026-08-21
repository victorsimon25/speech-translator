"""Adaptive backpressure controller (D28).

Policy, in order:
  1. Shrink: queue_depth >= shrink_trigger_depth for shrink_after_utterances
             consecutive utterances → cut effective max_utterance_ms by
             shrink_factor, floor at max_utterance_floor_ms.
  2. Recover: depth == 0 and RTF EMA < recover_rtf_ema for recover_after_s
              seconds → grow back.
  3. Drop: queue full → drop the oldest pending utterance, emit dropped event.
  4. Signal: any shrink or drop puts the session into degraded state; full
             recovery returns to running.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Callable

from ..config import Config


class BackpressureController:
    """Monitor the ASR queue and enforce D28's adaptive shrink-then-drop policy."""

    def __init__(
        self,
        asr_in_queue: Any,
        segmenter: Any,
        config: Config,
        on_drop: Callable[[str | None, int], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
    ) -> None:
        self._queue = asr_in_queue
        self._segmenter = segmenter
        self._config = config
        self._on_drop = on_drop
        self._on_state_change = on_state_change

        self._consecutive_deep = 0
        self._rtf_ema: float = 0.0
        self._recover_since: float | None = None
        self._degraded = False

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def try_enqueue(self, utterance: Any, last_sentence_id: str | None) -> dict | None:
        """Attempt to put utterance into the ASR queue.

        If the queue is full, drop the oldest entry and enqueue the new one.
        Returns a ``dropped`` event dict when a drop occurs, else None.

        Also tracks the queue depth for the shrink policy.
        """
        dropped_event: dict | None = None
        try:
            self._queue.put_nowait(utterance)
        except queue.Full:
            # Drop the oldest pending utterance (D28: stale caption has no value).
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(utterance)
            dropped_event = {
                "type": "dropped",
                "after_id": last_sentence_id,
                "utterances": 1,
            }
            if self._on_drop is not None:
                self._on_drop(last_sentence_id, 1)
            if not self._degraded:
                self._degraded = True
                if self._on_state_change is not None:
                    self._on_state_change("degraded")

        # Track depth for shrink policy.
        self.record_depth(self._queue.qsize())
        return dropped_event

    def record_depth(self, depth: int) -> None:
        """Track consecutive high-depth events; shrink when threshold is hit."""
        if depth >= self._config.shrink_trigger_depth:
            self._consecutive_deep += 1
            self._recover_since = None  # reset recovery window
        else:
            self._consecutive_deep = 0

        if self._consecutive_deep >= self._config.shrink_after_utterances:
            self._shrink()

    def update_rtf(self, rtf: float, depth: int, now: float) -> bool:
        """Update the RTF EMA and check whether recovery conditions are met.

        Returns True if the controller transitioned from degraded to running.
        """
        alpha = 0.3
        self._rtf_ema = alpha * rtf + (1.0 - alpha) * self._rtf_ema

        if not self._degraded:
            return False

        # Recovery requires depth == 0 and RTF EMA below the threshold for a
        # sustained window (D28: hysteresis prevents flapping).
        if depth == 0 and self._rtf_ema < self._config.recover_rtf_ema:
            if self._recover_since is None:
                self._recover_since = now
            elif now - self._recover_since >= self._config.recover_after_s:
                self._grow_back()
                self._recover_since = None
                return True
        else:
            self._recover_since = None
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rtf_ema(self) -> float:
        return self._rtf_ema

    @property
    def degraded(self) -> bool:
        return self._degraded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _shrink(self) -> None:
        current = self._segmenter.effective_max_utterance_ms
        new_ms = max(
            self._config.max_utterance_floor_ms,
            int(current * self._config.shrink_factor),
        )
        self._segmenter.set_max_utterance_ms(new_ms)
        self._consecutive_deep = 0

        if not self._degraded:
            self._degraded = True
            if self._on_state_change is not None:
                self._on_state_change("degraded")

    def _grow_back(self) -> None:
        current = self._segmenter.effective_max_utterance_ms
        # Inverse of shrink: divide by shrink_factor, ceiling at configured max.
        new_ms = min(
            self._config.max_utterance_ms,
            int(current / self._config.shrink_factor),
        )
        self._segmenter.set_max_utterance_ms(new_ms)

        # Only leave degraded when fully restored.
        if new_ms >= self._config.max_utterance_ms:
            self._degraded = False
            if self._on_state_change is not None:
                self._on_state_change("running")
