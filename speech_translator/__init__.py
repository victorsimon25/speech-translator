"""Real-time speech translator for online meetings.

Importing this package must succeed on Linux (D22): every Windows-only import
lives inside the function that needs it. Nothing here touches audio hardware,
CUDA, or the network.
"""

from __future__ import annotations

from .logging_setup import force_utf8

__version__ = "0.1.0"

# Done at import so no entry point — module, worker process, test — can forget
# it. The worker process is spawned on Windows and re-imports the package, so
# this covers it too (D27, D41).
force_utf8()

__all__ = ["__version__", "force_utf8"]
