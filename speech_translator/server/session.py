"""SessionController: asyncio state machine for the full pipeline (D27).

Six states: idle → loading → running ↔ degraded → idle | error

The controller owns the pipeline lifecycle: starts/stops the TranscribeWorker,
Segmenter, SentenceSplitter, run_translation_task, and Publisher. Every state
transition broadcasts {"type": "state", ...} to the connected WebSocket client.

The worker_factory parameter allows tests and the Linux dev loop to inject a
FakeTranscribeWorker without touching CUDA.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from ..config import Config, load_config
from ..segment.segmenter import Segmenter
from ..transcribe.protocol import WorkerError, WorkerResult
from ..transcribe.worker import TranscribeWorker
from ..translate.protocol import Sentence
from ..translate.splitter import SentenceSplitter
from ..translate.task import run_translation_task
from .backpressure import BackpressureController
from .publisher import Publisher

logger = logging.getLogger(__name__)


class SessionController:
    """Asyncio state machine for the pipeline.

    Usage::

        session = SessionController(config, worker_factory=FakeTranscribeWorker)
        session.set_broadcast(ws_send_json)
        await session.handle_message({"type": "start"})
        # ... pipeline runs ...
        await session.handle_message({"type": "stop"})
    """

    def __init__(
        self,
        config: Config | None = None,
        worker_factory: Callable[..., Any] | None = None,
        translator: Any | None = None,
    ) -> None:
        self._config = config or load_config()
        self._worker_factory = worker_factory or (lambda cfg=None: TranscribeWorker(cfg))
        self._translator_override = translator
        self._broadcast: Callable[[dict], Awaitable[None]] | None = None

        self._state = "idle"
        self._target_lang = self._config.target_lang
        self._source_lang_override: str | None = None

        # Set while pipeline is running.
        self._worker: Any | None = None
        self._segmenter: Segmenter | None = None
        self._splitter: SentenceSplitter | None = None
        self._backpressure: BackpressureController | None = None
        self._publisher: Publisher | None = None
        self._sentence_queue: asyncio.Queue | None = None
        self._translation_queue: asyncio.Queue | None = None

        # background asyncio tasks
        self._tasks: list[asyncio.Task] = []

        # utterance metadata for the bridge task (utterance_id → Utterance)
        self._pending_utterances: dict[str, Any] = {}
        self._last_sentence_id: str | None = None

        # LID accumulation
        self._lid_speech_ms: int = 0
        self._lid_locked = False

        # Audio thread stop event
        self._audio_stop: Any | None = None
        # Main asyncio event loop, captured in _start() for use from the audio thread.
        self._main_loop: Any | None = None

    # ------------------------------------------------------------------
    # Broadcast setter
    # ------------------------------------------------------------------

    def set_broadcast(self, fn: Callable[[dict], Awaitable[None]]) -> None:
        self._broadcast = fn

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    async def handle_message(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "start":
            if self._state == "idle":
                await self._start()
        elif t == "stop":
            if self._state not in ("idle", "error"):
                await self._stop()
        elif t == "set_target_lang":
            self._target_lang = msg.get("lang", self._target_lang)
        elif t == "override_source_lang":
            self._source_lang_override = msg.get("lang")

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def _set_state(self, new_state: str, detail: str = "") -> None:
        self._state = new_state
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "state", "state": new_state, "detail": detail})
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def _start(self) -> None:
        await self._set_state("loading", "Loading model (first run downloads weights)")

        worker = self._worker_factory(self._config)
        loop = asyncio.get_running_loop()
        self._main_loop = loop
        await loop.run_in_executor(None, worker.start)

        if worker.state == "error":
            err = worker.last_error
            detail = err.message if err is not None else "unknown error"
            await self._set_state("error", detail)
            return

        self._worker = worker

        # Build pipeline components.
        self._sentence_queue = asyncio.Queue()
        self._translation_queue = asyncio.Queue()

        self._segmenter = Segmenter(
            config=self._config,
            on_speaking=self._on_speaking,
        )

        self._splitter = SentenceSplitter()

        self._backpressure = BackpressureController(
            asr_in_queue=worker.in_queue,
            segmenter=self._segmenter,
            config=self._config,
            on_drop=self._on_drop,
            on_state_change=self._on_bp_state_change,
        )

        # Build translator (injected or real).
        if self._translator_override is not None:
            translator = self._translator_override
        else:
            from ..translate.translator import Translator
            api_key = self._config.google_translate_api_key or ""
            translator = Translator(
                api_key=api_key,
                target_lang=self._target_lang,
                config=self._config,
            )

        def budget_fn() -> tuple[int, int]:
            used = getattr(translator, "chars_used", 0)
            return used, self._config.mt_char_budget

        self._publisher = Publisher(
            send_fn=self._safe_broadcast,
            asr_in_queue=worker.in_queue,
            segmenter=self._segmenter,
            config=self._config,
            translator_budget_fn=budget_fn,
        )

        # Launch background tasks.
        self._tasks = [
            asyncio.create_task(
                self._asr_bridge(), name="asr_bridge"
            ),
            asyncio.create_task(
                run_translation_task(
                    self._sentence_queue,
                    self._translation_queue,
                    translator,
                ),
                name="translation",
            ),
            asyncio.create_task(
                self._publish_loop(), name="publish"
            ),
            asyncio.create_task(
                self._publisher.run_health_loop(), name="health"
            ),
        ]

        # Start audio capture on Windows (or when an audio source is available).
        # On Linux without hardware, we skip this; the pipeline still works for
        # any utterances the test/demo injects directly.
        try:
            from ..audio.select import open_source
            self._audio_stop = asyncio.Event()
            self._tasks.append(
                asyncio.create_task(
                    loop.run_in_executor(None, self._audio_thread), name="audio"
                )
            )
        except Exception:
            pass  # no audio source available on this platform — tests use the fake

        await self._set_state("running", "Pipeline active")

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    async def _stop(self) -> None:
        # Flush the sentence splitter before tearing down.
        if self._splitter is not None and self._publisher is not None and self._sentence_queue is not None:
            flushed = self._splitter.flush("silence", preceding_silence_ms=0)
            if flushed is not None:
                self._publisher.register_sentence(flushed)
                await self._sentence_queue.put(flushed)

        # Stop the translation task cleanly.
        if self._sentence_queue is not None:
            await self._sentence_queue.put(None)

        # Unblock _asr_bridge immediately: the bridge is awaiting out_queue.get
        # with a 0.5 s timeout; a None sentinel lets it exit without waiting.
        if self._worker is not None:
            try:
                self._worker.out_queue.put_nowait(None)
            except Exception:
                pass

        # Signal the audio thread to stop before cancelling asyncio tasks.
        if self._audio_stop is not None:
            self._audio_stop.set()

        # Cancel all background tasks.
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        # Stop the worker subprocess.
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception:
                pass
            self._worker = None

        if self._publisher is not None:
            self._publisher.close()

        self._segmenter = None
        self._splitter = None
        self._backpressure = None
        self._publisher = None
        self._sentence_queue = None
        self._translation_queue = None
        self._pending_utterances = {}
        self._lid_speech_ms = 0
        self._lid_locked = False
        self._last_sentence_id = None

        await self._set_state("idle", "")

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _asr_bridge(self) -> None:
        """Bridge the blocking multiprocessing out_queue to asyncio.

        Uses a 0.5 s timeout on queue.get so the thread never blocks
        indefinitely: task.cancel() is delivered within one timeout window,
        and asyncio.run() can shut down the executor cleanly.  A None sentinel
        on out_queue (put by _stop) causes an immediate exit.
        """
        loop = asyncio.get_running_loop()
        worker = self._worker  # capture before any potential None assignment
        while True:
            try:
                result = await loop.run_in_executor(
                    None, lambda: worker.out_queue.get(timeout=0.5)
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # queue.Empty from timeout — yield then retry
                await asyncio.sleep(0)
                continue

            if result is None:
                break

            if isinstance(result, WorkerResult):
                if result.transcript is not None:
                    await self._handle_transcript(result.transcript)

            elif isinstance(result, WorkerError):
                await self._set_state("error", result.message)
                break

    async def _handle_transcript(self, transcript: Any) -> None:
        utt = self._pending_utterances.pop(transcript.utterance_id, None)
        preceding = utt.preceding_silence_ms if utt is not None else 0
        closed_by = utt.closed_by if utt is not None else "silence"
        utt_duration_ms = (utt.end_ms - utt.start_ms) if utt is not None else 1

        sentences = self._splitter.feed(
            transcript,
            preceding_silence_ms=preceding,
            closed_by=closed_by,
        )
        for sentence in sentences:
            self._publisher.register_sentence(sentence)
            self._last_sentence_id = sentence.id
            await self._sentence_queue.put(sentence)

        # RTF and recovery.
        rtf = transcript.asr_ms / max(utt_duration_ms, 1)
        if self._backpressure is not None:
            self._backpressure.update_rtf(rtf, self._worker.in_queue.qsize(), time.time())
        self._publisher.update_rtf(rtf)

        # LID accumulation (D32).
        if not self._lid_locked and self._publisher is not None:
            self._lid_speech_ms += utt_duration_ms
            if self._lid_speech_ms >= self._config.lid_window_speech_ms:
                self._lid_locked = True
                await self._publisher.on_language_detected(
                    transcript.language, transcript.language_confidence
                )

        # Flush check: if silence > flush_silence_ms and we have a fragment.
        if (
            utt is not None
            and utt.preceding_silence_ms >= self._config.flush_silence_ms
            and self._splitter is not None
        ):
            flushed = self._splitter.flush("silence", preceding_silence_ms=utt.preceding_silence_ms)
            if flushed is not None and self._publisher is not None:
                self._publisher.register_sentence(flushed)
                await self._sentence_queue.put(flushed)

    async def _publish_loop(self) -> None:
        """Consume translations and forward them to the publisher."""
        while True:
            try:
                translation = await self._translation_queue.get()
            except asyncio.CancelledError:
                break
            if translation is None:
                break
            if self._publisher is not None:
                await self._publisher.on_translation(translation)
                self._publisher.update_mt_ms(translation.mt_ms)

    # ------------------------------------------------------------------
    # Audio thread (runs in executor, Windows only)
    # ------------------------------------------------------------------

    def _audio_thread(self) -> None:
        from ..audio.select import open_source
        loop = asyncio.new_event_loop()
        try:
            source = open_source()
            segmenter = self._segmenter
            for frame in source.frames():
                if self._audio_stop is not None and self._audio_stop.is_set():
                    break
                if segmenter is None:
                    break
                utterance = segmenter.push(frame)
                if utterance is not None:
                    self._pending_utterances[utterance.id] = utterance
                    if self._backpressure is not None:
                        dropped = self._backpressure.try_enqueue(utterance, self._last_sentence_id)
                        if dropped is not None and self._main_loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                self._safe_broadcast(dropped),
                                self._main_loop,
                            )
                    else:
                        try:
                            self._worker.in_queue.put_nowait(utterance)
                        except Exception:
                            pass
        except Exception as exc:
            logger.error("audio thread error: %s", exc)
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Callbacks and helpers
    # ------------------------------------------------------------------

    def _on_speaking(self, speaking: bool) -> None:
        msg = {"type": "vad", "speaking": speaking}
        if self._main_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._safe_broadcast(msg), self._main_loop)
            except Exception:
                pass

    def _on_drop(self, after_id: str | None, count: int) -> None:
        msg = {"type": "dropped", "after_id": after_id, "utterances": count}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.ensure_future(self._safe_broadcast(msg), loop=loop)

    def _on_bp_state_change(self, new_state: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.ensure_future(self._set_state(new_state, ""), loop=loop)

    async def _safe_broadcast(self, msg: dict) -> None:
        if self._broadcast is not None:
            try:
                await self._broadcast(msg)
            except Exception:
                pass
