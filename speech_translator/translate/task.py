"""Single serialised translation task (D27)."""

from __future__ import annotations

import asyncio

from .protocol import Sentence, Translation
from .translator import TranslatorProtocol


async def run_translation_task(
    sentence_queue: asyncio.Queue,
    translation_queue: asyncio.Queue,
    translator: TranslatorProtocol,
) -> None:
    """Consume sentences from sentence_queue and put translations into translation_queue.

    A single task guarantees FIFO ordering without a reorder buffer (D27).
    Stops cleanly on sentinel None.
    """
    while True:
        item: Sentence | None = await sentence_queue.get()
        if item is None:
            break
        translation: Translation = await translator.translate(item)
        await translation_queue.put(translation)
