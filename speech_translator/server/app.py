"""FastAPI application: static UI + WebSocket endpoint (D38).

One WebSocket client at a time — this is a demo tool, not a multi-user server
(D38). A second connection is refused with code 1008.

Usage::

    app = create_app(config, worker_factory=FakeTranscribeWorker)
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config, load_config
from .session import SessionController

STATIC_DIR = pathlib.Path(__file__).parent / "static"


def create_app(
    config: Config | None = None,
    worker_factory: Callable[..., Any] | None = None,
    translator: Any | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters are injectable so tests can pass FakeTranscribeWorker and skip
    CUDA/audio dependencies entirely.
    """
    cfg = config or load_config()
    session = SessionController(
        config=cfg,
        worker_factory=worker_factory,
        translator=translator,
    )

    app = FastAPI(title="Speech Translator")

    # Single-client slot — set when a WebSocket is connected, cleared on disconnect.
    _client: WebSocket | None = None

    async def broadcast(msg: dict) -> None:
        if _client is not None:
            try:
                await _client.send_json(msg)
            except Exception:
                pass

    session.set_broadcast(broadcast)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        nonlocal _client
        await ws.accept()

        if _client is not None:
            await ws.send_json({
                "type": "error",
                "detail": "Only one client at a time (D38)",
            })
            await ws.close(1008)
            return

        _client = ws
        # Re-set broadcast now that _client is assigned, so the first messages
        # (including the initial state broadcast if any) reach this client.
        session.set_broadcast(broadcast)

        try:
            while True:
                data = await ws.receive_json()
                await session.handle_message(data)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            _client = None

    return app
