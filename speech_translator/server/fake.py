"""FakeTranscribeWorker — drop-in for TranscribeWorker with no GPU, no subprocess.

Used by SessionController(worker_factory=...) for tests and the Linux dev loop.
The fake starts instantly (no model download), processes utterances from a
background daemon thread, and returns pre-canned or synthesized transcripts.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from ..config import Config, load_config
from ..transcribe.protocol import Transcript, WorkerResult


class FakeTranscribeWorker:
    """Synchronous, in-process stand-in for TranscribeWorker.

    Satisfies the same interface: start() / stop(), .in_queue, .out_queue,
    .state, .last_error.  start() returns immediately and does NOT put a
    WorkerReady sentinel on out_queue — the SessionController infers readiness
    from worker.state, not by draining out_queue (unlike the real worker whose
    start() drains its own out_queue before returning).
    """

    def __init__(
        self,
        config: Config | None = None,
        canned_transcripts: list[Transcript] | None = None,
    ) -> None:
        self._config = config or load_config()
        self._canned = canned_transcripts
        self._in_q: queue.Queue = queue.Queue(maxsize=self._config.asr_queue_maxsize)
        self._out_q: queue.Queue = queue.Queue()
        self._state = "idle"
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # TranscribeWorker interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Set state to ready and begin processing in a background thread.

        Does NOT put WorkerReady on out_queue — the real TranscribeWorker's
        start() drains WorkerReady internally, so the bridge task never sees
        it.  We match that contract here.
        """
        self._state = "ready"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._in_q.put(None)  # shutdown sentinel for worker thread
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._state = "idle"

    @property
    def in_queue(self) -> Any:
        return self._in_q

    @property
    def out_queue(self) -> Any:
        return self._out_q

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_error(self) -> Any:
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        idx = 0
        while True:
            item = self._in_q.get()
            if item is None:
                break
            utterance = item
            if self._canned:
                t = self._canned[idx % len(self._canned)]
                transcript = Transcript(
                    utterance_id=utterance.id,
                    text=t.text,
                    language=t.language,
                    language_confidence=t.language_confidence,
                    asr_ms=t.asr_ms,
                    no_speech_prob=t.no_speech_prob,
                    avg_logprob=t.avg_logprob,
                )
                idx += 1
            else:
                transcript = Transcript(
                    utterance_id=utterance.id,
                    text=f"Fake transcription for {utterance.id}.",
                    language="en",
                    language_confidence=0.99,
                    asr_ms=50,
                    no_speech_prob=0.01,
                    avg_logprob=-0.3,
                )
            self._out_q.put(WorkerResult(transcript=transcript))
