"""Tests for Level 6: server, backpressure, publisher, session, WebSocket.

All six tests run on Linux with no GPU, no audio device, and no network.
"""

from __future__ import annotations

import asyncio
import queue

import pytest

from speech_translator.config import Config
from speech_translator.segment.segmenter import Segmenter
from speech_translator.segment.segmenter import Utterance as UttType
from speech_translator.server.backpressure import BackpressureController
from speech_translator.server.fake import FakeTranscribeWorker
from speech_translator.server.publisher import Publisher
from speech_translator.server.session import SessionController
from speech_translator.translate.protocol import Sentence, Translation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_utterance(uid: str, duration_ms: int = 1000) -> UttType:
    """Synthetic utterance with `duration_ms` ms of silent PCM."""
    pcm = b"\x00" * (duration_ms * 32)  # 16 kHz mono int16 = 32 bytes/ms
    return UttType(
        id=uid,
        pcm=pcm,
        start_ms=0,
        end_ms=duration_ms,
        closed_by="silence",
        preceding_silence_ms=0,
    )


def _make_sentence(
    sid: str,
    uid: str,
    text: str = "Hello.",
    lang: str = "en",
    preceding_silence_ms: int = 0,
    closed_by: str = "silence",
) -> Sentence:
    return Sentence(
        id=sid,
        utterance_id=uid,
        text=text,
        language=lang,
        preceding_silence_ms=preceding_silence_ms,
        closed_by=closed_by,  # type: ignore[arg-type]
        is_flush=False,
    )


def _make_translation(
    sid: str,
    source: str = "Hello.",
    target: str = "Hola.",
    source_lang: str = "en",
    target_lang: str = "es",
    mt_ms: int = 150,
) -> Translation:
    return Translation(
        sentence_id=sid,
        source_text=source,
        target_text=target,
        source_lang=source_lang,
        target_lang=target_lang,
        mt_ms=mt_ms,
        mt_error=None,
    )


# ---------------------------------------------------------------------------
# 1. State machine transitions
# ---------------------------------------------------------------------------


def test_state_machine_transitions() -> None:
    """idle → loading → running → idle via start/stop."""
    states: list[str] = []

    async def _body() -> None:
        cfg = Config()
        session = SessionController(
            config=cfg,
            worker_factory=lambda _cfg=None: FakeTranscribeWorker(_cfg or cfg),
        )

        async def capture(msg: dict) -> None:
            if msg.get("type") == "state":
                states.append(msg["state"])

        session.set_broadcast(capture)

        await session.handle_message({"type": "start"})
        # Give background tasks a moment to start.
        await asyncio.sleep(0.05)
        await session.handle_message({"type": "stop"})
        await asyncio.sleep(0.05)

    asyncio.run(_body())

    assert "loading" in states, f"expected 'loading' in {states}"
    assert "running" in states, f"expected 'running' in {states}"
    assert states[-1] == "idle", f"expected final state 'idle', got {states}"


# ---------------------------------------------------------------------------
# 2. Caption join
# ---------------------------------------------------------------------------


def test_caption_join() -> None:
    """Publisher correctly joins Translation + Sentence by sentence_id.
    preceding_silence_ms and closed_by must come from the Sentence side."""
    captions: list[dict] = []

    async def _body() -> None:
        async def send_fn(msg: dict) -> None:
            if msg["type"] == "caption":
                captions.append(msg)

        cfg = Config()
        seg = Segmenter(cfg)
        asr_q: queue.Queue = queue.Queue(maxsize=4)

        pub = Publisher(
            send_fn=send_fn,
            asr_in_queue=asr_q,
            segmenter=seg,
            config=cfg,
            translator_budget_fn=lambda: (0, cfg.mt_char_budget),
        )

        sentence = _make_sentence(
            "u_0001.s1", "u_0001",
            text="¿Me escuchan bien?",
            lang="es",
            preceding_silence_ms=1240,
            closed_by="silence",
        )
        translation = _make_translation(
            "u_0001.s1",
            source="¿Me escuchan bien?",
            target="Can you hear me okay?",
            source_lang="es",
            target_lang="en",
            mt_ms=200,
        )

        pub.register_sentence(sentence)
        await pub.on_translation(translation)

    asyncio.run(_body())

    assert len(captions) == 1, f"expected 1 caption, got {len(captions)}"
    c = captions[0]
    assert c["type"] == "caption"
    assert c["id"] == "u_0001.s1"
    assert c["utterance_id"] == "u_0001"
    assert c["source_text"] == "¿Me escuchan bien?"
    assert c["target_text"] == "Can you hear me okay?"
    assert c["preceding_silence_ms"] == 1240, f"got {c['preceding_silence_ms']}"
    assert c["closed_by"] == "silence"
    assert "latency_ms" in c
    assert c["latency_ms"]["mt"] == 200


# ---------------------------------------------------------------------------
# 3. Backpressure shrink
# ---------------------------------------------------------------------------


def test_backpressure_shrink() -> None:
    """Queue depth >= threshold for N consecutive utterances → effective_max_utterance_ms shrinks."""
    cfg = Config()
    seg = Segmenter(cfg)
    original_max = seg.effective_max_utterance_ms

    # Use a small queue so we can fill it easily.
    asr_q: queue.Queue = queue.Queue(maxsize=cfg.asr_queue_maxsize)

    bp = BackpressureController(
        asr_in_queue=asr_q,
        segmenter=seg,
        config=cfg,
    )

    # Pre-fill the queue so qsize() >= shrink_trigger_depth after each enqueue.
    for i in range(cfg.shrink_trigger_depth):
        asr_q.put(_make_utterance(f"u_{i:04d}"))

    # Enqueue utterances — each call records depth and increments consecutive counter.
    for i in range(cfg.shrink_after_utterances):
        # Put one more then consume (to avoid Full) so depth stays at trigger level.
        if not asr_q.full():
            bp.try_enqueue(_make_utterance(f"u_fill_{i}"), None)
        else:
            bp.record_depth(cfg.shrink_trigger_depth)

    assert seg.effective_max_utterance_ms < original_max, (
        f"expected shrink: {seg.effective_max_utterance_ms} < {original_max}"
    )
    assert bp.degraded


# ---------------------------------------------------------------------------
# 4. Backpressure drop oldest
# ---------------------------------------------------------------------------


def test_backpressure_drop_oldest() -> None:
    """Full queue → oldest item dropped, dropped event emitted."""
    cfg = Config()
    seg = Segmenter(cfg)

    asr_q: queue.Queue = queue.Queue(maxsize=2)

    dropped_events: list[dict] = []

    def on_drop(after_id: str | None, count: int) -> None:
        dropped_events.append({"after_id": after_id, "utterances": count})

    bp = BackpressureController(
        asr_in_queue=asr_q,
        segmenter=seg,
        config=cfg,
        on_drop=on_drop,
    )

    u1 = _make_utterance("u_0001")
    u2 = _make_utterance("u_0002")
    u3 = _make_utterance("u_0003")

    # Fill the queue to capacity.
    asr_q.put_nowait(u1)
    asr_q.put_nowait(u2)

    assert asr_q.full()

    # This should drop u1 (oldest) and enqueue u3.
    result = bp.try_enqueue(u3, "u_0001.s1")

    assert result is not None, "expected a dropped event dict"
    assert result["type"] == "dropped"
    assert result["after_id"] == "u_0001.s1"
    assert result["utterances"] == 1

    assert len(dropped_events) == 1
    assert dropped_events[0]["after_id"] == "u_0001.s1"

    # Queue should still be at capacity with u2 and u3.
    assert asr_q.qsize() == 2
    items = [asr_q.get_nowait(), asr_q.get_nowait()]
    ids = {item.id for item in items}
    assert "u_0001" not in ids, "u_0001 (oldest) should have been dropped"
    assert "u_0002" in ids
    assert "u_0003" in ids

    assert bp.degraded


# ---------------------------------------------------------------------------
# 5. WebSocket state broadcast
# ---------------------------------------------------------------------------


def test_websocket_state_broadcast() -> None:
    """State transitions are broadcast over the WebSocket."""
    from starlette.testclient import TestClient

    from speech_translator.server.app import create_app

    cfg = Config()
    app = create_app(
        config=cfg,
        worker_factory=lambda _cfg=None: FakeTranscribeWorker(_cfg or cfg),
    )

    messages: list[dict] = []

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start"})

            # Collect the first few messages — at minimum we expect state:loading.
            for _ in range(5):
                try:
                    msg = ws.receive_json()
                    messages.append(msg)
                    # Stop collecting once we see running (or after enough msgs).
                    if msg.get("type") == "state" and msg.get("state") == "running":
                        break
                except Exception:
                    break

            ws.send_json({"type": "stop"})

    state_msgs = [m for m in messages if m.get("type") == "state"]
    state_names = [m["state"] for m in state_msgs]

    assert "loading" in state_names, (
        f"expected 'loading' in broadcast states, got {state_names}"
    )


# ---------------------------------------------------------------------------
# 6. Health message fields
# ---------------------------------------------------------------------------


def test_health_message_fields() -> None:
    """health message contains all required fields with correct types."""
    received: list[dict] = []

    async def _body() -> None:
        async def send_fn(msg: dict) -> None:
            received.append(msg)

        cfg = Config()
        seg = Segmenter(cfg)
        asr_q: queue.Queue = queue.Queue(maxsize=4)

        pub = Publisher(
            send_fn=send_fn,
            asr_in_queue=asr_q,
            segmenter=seg,
            config=cfg,
            translator_budget_fn=lambda: (12840, cfg.mt_char_budget),
        )

        await pub._send_health()

    asyncio.run(_body())

    assert len(received) == 1
    h = received[0]

    assert h.get("type") == "health"
    assert isinstance(h.get("queue_depth"), int)
    assert isinstance(h.get("rtf"), float)
    assert isinstance(h.get("word_age_ms"), (int, float))
    assert isinstance(h.get("effective_max_utterance_ms"), int)
    assert isinstance(h.get("chars_used"), int)
    assert isinstance(h.get("chars_budget"), int)
    assert h["chars_used"] == 12840
    assert h["chars_budget"] == 500_000
