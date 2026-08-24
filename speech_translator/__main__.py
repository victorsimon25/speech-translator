"""Application entry point — ``python -m speech_translator``.

Starts the FastAPI + uvicorn server on host:port from config (D38).
Set DEMO_MODE=1 to use FakeTranscribeWorker — no GPU or audio required.
This is the default in the Docker image (D72).
"""

from __future__ import annotations

import os
import sys

import uvicorn

from .config import load_config
from .logging_setup import force_utf8
from .server.app import create_app


def main() -> int:
    force_utf8()
    cfg = load_config()

    worker_factory = None  # defaults to real TranscribeWorker
    if os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes"):
        from .server.fake import FakeTranscribeWorker
        worker_factory = FakeTranscribeWorker
        print(
            "speech-translator [DEMO_MODE] — FakeTranscribeWorker active, no GPU or audio required",
            file=sys.stderr,
        )

    app = create_app(cfg, worker_factory=worker_factory)
    print(
        f"speech-translator starting on http://{cfg.host}:{cfg.port}",
        file=sys.stderr,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
