"""Application entry point — ``python -m speech_translator``.

Starts the FastAPI + uvicorn server on host:port from config (D38).
The fake ASR path (FakeTranscribeWorker) is the default on Linux when no
CUDA device is present, so the full pipeline can be exercised without the
demo machine.
"""

from __future__ import annotations

import sys

import uvicorn

from .config import load_config
from .logging_setup import force_utf8
from .server.app import create_app


def main() -> int:
    force_utf8()
    cfg = load_config()
    app = create_app(cfg)
    print(
        f"speech-translator starting on http://{cfg.host}:{cfg.port}",
        file=sys.stderr,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
