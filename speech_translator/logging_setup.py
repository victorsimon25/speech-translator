"""UTF-8 enforcement and logging setup.

The Windows console is cp1252 and this application's entire output — captions in
Spanish, Japanese, Arabic — is non-ASCII. Printing one caption to an
unreconfigured stream raises ``UnicodeEncodeError`` and kills the thread it
happens on. That is D41's most likely "works on my machine" failure, so UTF-8 is
forced at import of the package rather than left to each entry point to
remember.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

_UTF8_FORCED = False


def force_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8. Idempotent, safe to call anywhere.

    ``errors="replace"`` rather than ``strict``: a console that genuinely cannot
    render a glyph should show a replacement character, not take the process
    down with it.
    """
    global _UTF8_FORCED
    if _UTF8_FORCED:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Redirected to something that cannot be reconfigured (a pipe
                # opened in binary mode, a pytest capture object). Not fatal.
                pass
    _UTF8_FORCED = True


def stream_is_utf8(stream: TextIO) -> bool:
    return (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8"


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure root logging with UTF-8 on every handler (D41)."""
    force_utf8()

    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stderr)
    handlers.append(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # encoding is explicit: FileHandler defaults to locale encoding, which
        # on the Windows demo machine is cp1252.
        handlers.append(logging.FileHandler(log_file, encoding="utf-8", errors="replace"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
