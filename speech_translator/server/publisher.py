"""Publisher: join Translations back to their Sentences and push to the WebSocket.

The Publisher owns the Sentence registry — a dict from sentence_id to Sentence.
When a Translation arrives (via on_translation), it pops the matching Sentence,
combines the fields that the Translator does not carry (utterance_id,
preceding_silence_ms, closed_by), and broadcasts a caption message.

It also emits health every 2 seconds and language_detected events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from ..config import Config
from ..translate.protocol import Sentence, Translation

logger = logging.getLogger(__name__)

_EMA_ALPHA = 0.3


class Publisher:
    """Joins Translations to their Sentences and emits WebSocket messages."""

    def __init__(
        self,
        send_fn: Callable[[dict], Awaitable[None]],
        asr_in_queue: Any,
        segmenter: Any,
        config: Config,
        translator_budget_fn: Callable[[], tuple[int, int]],
    ) -> None:
        self._send = send_fn
        self._queue = asr_in_queue
        self._segmenter = segmenter
        self._config = config
        self._budget_fn = translator_budget_fn

        self._sentences: dict[str, Sentence] = {}
        self._rtf_ema: float = 0.0
        self._mt_ms_avg: float = 300.0  # initial estimate; updated on each translation

    # ------------------------------------------------------------------
    # Sentence registry
    # ------------------------------------------------------------------

    def register_sentence(self, sentence: Sentence) -> None:
        """Register a Sentence before it is dispatched to the translator."""
        self._sentences[sentence.id] = sentence

    # ------------------------------------------------------------------
    # Incoming events
    # ------------------------------------------------------------------

    async def on_translation(self, translation: Translation) -> None:
        """Join a Translation to its Sentence and emit a caption message."""
        sentence = self._sentences.pop(translation.sentence_id, None)
        if sentence is None:
            logger.warning("no sentence for id %s — caption dropped", translation.sentence_id)
            return

        word_age = (
            self._config.silence_threshold_ms
            + self._mt_ms_avg
            + self._segmenter.effective_max_utterance_ms * (1.0 + self._rtf_ema)
        )

        caption: dict = {
            "type": "caption",
            "id": translation.sentence_id,
            "utterance_id": sentence.utterance_id,
            "source_text": translation.source_text,
            "target_text": translation.target_text,
            "mt_error": translation.mt_error,
            "source_lang": translation.source_lang,
            "target_lang": translation.target_lang,
            "preceding_silence_ms": sentence.preceding_silence_ms,
            "closed_by": sentence.closed_by,
            "latency_ms": {
                "asr": 0,
                "mt": translation.mt_ms,
                "word_age": round(word_age),
            },
        }
        await self._send(caption)

    async def on_language_detected(self, lang: str, confidence: float) -> None:
        await self._send({"type": "language_detected", "lang": lang, "confidence": confidence})

    # ------------------------------------------------------------------
    # RTF / latency tracking
    # ------------------------------------------------------------------

    def update_rtf(self, rtf: float) -> None:
        self._rtf_ema = _EMA_ALPHA * rtf + (1.0 - _EMA_ALPHA) * self._rtf_ema

    def update_mt_ms(self, mt_ms: int) -> None:
        if mt_ms > 0:
            self._mt_ms_avg = _EMA_ALPHA * mt_ms + (1.0 - _EMA_ALPHA) * self._mt_ms_avg

    # ------------------------------------------------------------------
    # Health loop
    # ------------------------------------------------------------------

    async def run_health_loop(self) -> None:
        """Emit a health message every 2 seconds until cancelled."""
        while True:
            await asyncio.sleep(2)
            await self._send_health()

    async def _send_health(self) -> None:
        chars_used, chars_budget = self._budget_fn()
        effective_max = self._segmenter.effective_max_utterance_ms
        word_age_ms = (
            self._config.silence_threshold_ms
            + self._mt_ms_avg
            + effective_max * (1.0 + self._rtf_ema)
        )
        await self._send({
            "type": "health",
            "queue_depth": self._queue.qsize(),
            "rtf": round(self._rtf_ema, 3),
            "word_age_ms": round(word_age_ms),
            "effective_max_utterance_ms": effective_max,
            "chars_used": chars_used,
            "chars_budget": chars_budget,
        })
