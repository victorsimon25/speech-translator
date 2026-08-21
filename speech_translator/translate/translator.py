"""LibreTranslate REST client (INTERFACES.md §5, D31, D68)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx

from ..config import Config, load_config
from .protocol import Sentence, Translation

logger = logging.getLogger(__name__)


class _LRUCache:
    """OrderedDict-based LRU cache for translation results."""

    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: object) -> str | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: object, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = value


class TranslatorProtocol(Protocol):
    """Structural interface consumed by run_translation_task and tests."""

    async def translate(self, sentence: Sentence) -> Translation:
        ...


class Translator:
    """Translate sentences via LibreTranslate REST API (D68).

    Budget tracking (D31): cumulative characters are persisted to a month-keyed
    JSON file under config.data_dir / "usage" so that a process restart cannot
    reset the counter.  The httpx.AsyncClient is created once and reused.
    """

    def __init__(
        self,
        api_key: str,
        target_lang: str,
        cache_maxsize: int = 512,
        config: Config | None = None,
    ) -> None:
        self._api_key = api_key
        self._target_lang = target_lang
        self._config = config if config is not None else load_config()
        self._cache = _LRUCache(cache_maxsize)
        self._client = httpx.AsyncClient(timeout=self._config.translate_timeout_s)

        month_key = datetime.now().strftime("%Y_%m")
        self._budget_file: Path = (
            self._config.data_dir / "usage" / f"chars_{month_key}.json"
        )
        self._chars_used: int = self._load_budget()

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def _load_budget(self) -> int:
        if not self._budget_file.exists():
            return 0
        try:
            data = json.loads(self._budget_file.read_text(encoding="utf-8"))
            return int(data.get("chars_used", 0))
        except (json.JSONDecodeError, ValueError, OSError):
            return 0

    def _save_budget(self) -> None:
        self._budget_file.parent.mkdir(parents=True, exist_ok=True)
        self._budget_file.write_text(
            json.dumps({"chars_used": self._chars_used}), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def chars_used(self) -> int:
        """Characters consumed this month (D31); exposed for the health message."""
        return self._chars_used

    async def translate(self, sentence: Sentence) -> Translation:
        src = sentence.language
        tgt = self._target_lang
        text = sentence.text

        # Skip when source and target are the same language.
        if src == tgt:
            return Translation(
                sentence_id=sentence.id,
                source_text=text,
                target_text=text,
                source_lang=src,
                target_lang=tgt,
                mt_ms=0,
                mt_error=None,
            )

        # Hard-stop when the monthly ceiling has been reached.
        if self._chars_used >= self._config.mt_char_hard_stop:
            return Translation(
                sentence_id=sentence.id,
                source_text=text,
                target_text=None,
                source_lang=src,
                target_lang=tgt,
                mt_ms=0,
                mt_error="budget_exhausted",
            )

        # LRU cache hit — free characters and free latency.
        cache_key = (src, tgt, text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return Translation(
                sentence_id=sentence.id,
                source_text=text,
                target_text=cached,
                source_lang=src,
                target_lang=tgt,
                mt_ms=0,
                mt_error=None,
            )

        # Warn when approaching the budget ceiling.
        if self._chars_used / self._config.mt_char_budget > self._config.mt_char_warn_ratio:
            logger.warning(
                "MT character budget %.0f%% used (%d / %d)",
                100.0 * self._chars_used / self._config.mt_char_budget,
                self._chars_used,
                self._config.mt_char_budget,
            )

        # HTTP call with one retry and 200 ms backoff.
        t0 = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self._config.translate_retries):
            if attempt > 0:
                await asyncio.sleep(0.2)
            try:
                params: dict = {"q": text, "langpair": f"{src}|{tgt}"}
                if self._api_key:
                    params["de"] = self._api_key  # email raises daily quota to 10 000 words
                resp = await self._client.get(
                    self._config.translate_url,
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("quotaFinished"):
                    raise RuntimeError("MyMemory daily quota reached")
                translated: str = data["responseData"]["translatedText"]
                mt_ms = int((time.monotonic() - t0) * 1000)

                self._cache.set(cache_key, translated)
                self._chars_used += len(text)
                self._save_budget()

                return Translation(
                    sentence_id=sentence.id,
                    source_text=text,
                    target_text=translated,
                    source_lang=src,
                    target_lang=tgt,
                    mt_ms=mt_ms,
                    mt_error=None,
                )
            except Exception as exc:
                last_error = exc

        mt_ms = int((time.monotonic() - t0) * 1000)
        return Translation(
            sentence_id=sentence.id,
            source_text=text,
            target_text=None,
            source_lang=src,
            target_lang=tgt,
            mt_ms=mt_ms,
            mt_error=str(last_error),
        )
