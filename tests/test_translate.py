"""Tests for Level 5: SentenceSplitter, Translator, run_translation_task.

All tests run on Linux with no network, no GPU, and no audio device.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from speech_translator.config import Config
from speech_translator.transcribe.protocol import Transcript
from speech_translator.translate.protocol import Sentence, Translation
from speech_translator.translate.splitter import SentenceSplitter
from speech_translator.translate.task import run_translation_task
from speech_translator.translate.translator import Translator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transcript(
    uid: str,
    text: str,
    language: str = "en",
) -> Transcript:
    return Transcript(
        utterance_id=uid,
        text=text,
        language=language,
        language_confidence=0.99,
        asr_ms=100,
        no_speech_prob=0.01,
        avg_logprob=-0.3,
    )


def _make_sentence(
    sid: str,
    uid: str,
    text: str,
    language: str = "en",
    preceding_silence_ms: int = 0,
    closed_by: str = "silence",
    is_flush: bool = False,
) -> Sentence:
    return Sentence(
        id=sid,
        utterance_id=uid,
        text=text,
        language=language,
        preceding_silence_ms=preceding_silence_ms,
        closed_by=closed_by,  # type: ignore[arg-type]
        is_flush=is_flush,
    )


class _FakeTranslator:
    """Returns translations instantly, echoing source text."""

    async def translate(self, sentence: Sentence) -> Translation:
        return Translation(
            sentence_id=sentence.id,
            source_text=sentence.text,
            target_text=sentence.text,
            source_lang=sentence.language,
            target_lang="en",
            mt_ms=0,
            mt_error=None,
        )


# ---------------------------------------------------------------------------
# SentenceSplitter tests
# ---------------------------------------------------------------------------


def test_splitter_carry_over() -> None:
    """Fragment from a max-length utterance prepends to the next and forms a complete sentence."""
    splitter = SentenceSplitter()

    # First utterance ends mid-sentence — no trailing punctuation.
    sentences = splitter.feed(
        _make_transcript("u1", "Hello world", "en"),
        preceding_silence_ms=0,
        closed_by="max_length",
    )
    assert sentences == [], "no complete sentence yet — fragment held"

    # Second utterance completes the sentence.
    sentences = splitter.feed(
        _make_transcript("u2", "how are you?", "en"),
        preceding_silence_ms=0,
        closed_by="silence",
    )
    assert len(sentences) == 1
    assert sentences[0].text == "Hello world how are you?"
    assert sentences[0].utterance_id == "u2"
    assert sentences[0].id == "u2.s1"


def test_splitter_flush_emits_held_fragment() -> None:
    """flush() emits the held fragment with is_flush=True; a second call returns None."""
    splitter = SentenceSplitter()

    sentences = splitter.feed(
        _make_transcript("u1", "Still talking", "en"),
        preceding_silence_ms=0,
        closed_by="silence",
    )
    assert sentences == []

    flushed = splitter.flush("silence", preceding_silence_ms=1200)
    assert flushed is not None
    assert flushed.text == "Still talking"
    assert flushed.is_flush is True
    assert flushed.utterance_id == "u1"

    # Nothing left to flush.
    assert splitter.flush("silence", preceding_silence_ms=0) is None


def test_splitter_preceding_silence_ms() -> None:
    """Only the first sentence of each utterance carries preceding_silence_ms; the rest carry 0."""
    splitter = SentenceSplitter()

    sentences = splitter.feed(
        _make_transcript("u1", "First. Second. Third.", "en"),
        preceding_silence_ms=500,
        closed_by="silence",
    )
    assert len(sentences) == 3, f"expected 3 sentences, got: {[s.text for s in sentences]}"
    assert sentences[0].preceding_silence_ms == 500
    assert sentences[1].preceding_silence_ms == 0
    assert sentences[2].preceding_silence_ms == 0


# ---------------------------------------------------------------------------
# run_translation_task ordering test
# ---------------------------------------------------------------------------


def test_translation_ordering() -> None:
    """Translations emerge from run_translation_task in the same order as input sentences."""

    async def _body() -> None:
        sentence_q: asyncio.Queue = asyncio.Queue()
        translation_q: asyncio.Queue = asyncio.Queue()

        sentences = [
            _make_sentence(f"u1.s{i}", "u1", f"Sentence {i}.", "en")
            for i in range(1, 6)
        ]
        for s in sentences:
            await sentence_q.put(s)
        await sentence_q.put(None)  # sentinel

        await run_translation_task(sentence_q, translation_q, _FakeTranslator())

        results = []
        while not translation_q.empty():
            results.append(await translation_q.get())

        assert [r.sentence_id for r in results] == [s.id for s in sentences]

    asyncio.run(_body())


# ---------------------------------------------------------------------------
# Translator tests
# ---------------------------------------------------------------------------


def test_translation_budget_persists(tmp_path: Path) -> None:
    """A pre-existing budget file with chars_used >= mt_char_hard_stop causes budget_exhausted."""
    month_key = datetime.now().strftime("%Y_%m")
    budget_dir = tmp_path / "usage"
    budget_dir.mkdir()
    (budget_dir / f"chars_{month_key}.json").write_text(
        json.dumps({"chars_used": 480_000}), encoding="utf-8"
    )

    cfg = Config(data_dir=tmp_path)
    assert cfg.mt_char_hard_stop <= 480_000, "test assumption: 480k must hit the hard stop"

    translator = Translator(api_key="fake-key", target_lang="en", config=cfg)
    sentence = _make_sentence("u1.s1", "u1", "Bonjour le monde.", "fr")

    async def _body() -> None:
        result = await translator.translate(sentence)
        assert result.mt_error == "budget_exhausted"
        assert result.target_text is None

    asyncio.run(_body())


def test_translation_api_failure_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the HTTP call always raises, translate() returns target_text=None with mt_error set."""

    async def _always_fail(self_client: object, url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr(httpx.AsyncClient, "post", _always_fail)

    cfg = Config(data_dir=tmp_path)
    translator = Translator(api_key="fake-key", target_lang="en", config=cfg)
    sentence = _make_sentence("u1.s1", "u1", "Bonjour.", "fr")

    async def _body() -> None:
        result = await translator.translate(sentence)
        assert result.target_text is None
        assert result.mt_error is not None
        assert "simulated network failure" in result.mt_error

    asyncio.run(_body())


def test_translation_skip_same_lang(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When source_lang == target_lang, target_text equals source_text and no HTTP call is made."""
    called: list[int] = []

    async def _should_not_be_called(self_client: object, url: str, **kwargs: object) -> None:
        called.append(1)
        raise AssertionError("HTTP must not be called for same-language pair")

    monkeypatch.setattr(httpx.AsyncClient, "post", _should_not_be_called)

    cfg = Config(data_dir=tmp_path)
    translator = Translator(api_key="fake-key", target_lang="en", config=cfg)
    sentence = _make_sentence("u1.s1", "u1", "Hello world.", "en")  # source == target

    async def _body() -> None:
        result = await translator.translate(sentence)
        assert result.target_text == result.source_text
        assert result.mt_error is None
        assert not called

    asyncio.run(_body())


def test_translation_lru_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Translating the same text twice hits the HTTP endpoint exactly once."""
    call_count: list[int] = []

    async def _fake_post(self_client: object, url: str, **kwargs: object) -> object:
        call_count.append(1)

        class _FakeResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"data": {"translations": [{"translatedText": "Hola mundo."}]}}

        return _FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    cfg = Config(data_dir=tmp_path)
    translator = Translator(api_key="fake-key", target_lang="en", config=cfg)

    # Same text, different sentence IDs (simulating repeated utterance content).
    s1 = _make_sentence("u1.s1", "u1", "Hello world.", "fr")
    s2 = _make_sentence("u2.s1", "u2", "Hello world.", "fr")

    async def _body() -> None:
        r1 = await translator.translate(s1)
        r2 = await translator.translate(s2)
        assert len(call_count) == 1, "cache should prevent second HTTP call"
        assert r1.target_text == r2.target_text == "Hola mundo."

    asyncio.run(_body())
