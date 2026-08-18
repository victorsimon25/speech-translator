"""Application entry point — ``python -m speech_translator``.

Not built yet. The server and UI land at Level 6 (`docs/PLAN.md`); this exists
so the documented command names what is missing instead of raising a bare
``No module named speech_translator.__main__``.
"""

from __future__ import annotations

import sys

from . import __version__
from .logging_setup import force_utf8


def main() -> int:
    force_utf8()
    print(
        f"speech-translator {__version__} — the pipeline is not built yet.\n"
        f"Level 0 (foundation) is complete; the server and UI arrive at Level 6.\n"
        f"See docs/PLAN.md for the ladder, and run the preflight now:\n"
        f"\n"
        f"    python -m speech_translator.doctor\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
