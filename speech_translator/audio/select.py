"""The one place platform detection is allowed to appear.

`INTERFACES.md` §1: "Source selection happens once at startup and is the *only*
place platform detection is allowed to appear." This is that place. Everything
else in the codebase takes an `AudioSource` and does not know or care which one
it got.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .source import AudioSource, AudioDeviceUnavailable
from .wav_source import WavFileSource


def open_source(
    wav: str | Path | None = None,
    device_index: int | None = None,
    realtime: bool = True,
) -> AudioSource:
    """Pick a source: an explicit WAV, else the platform's live capture.

    `PipeWireMonitorSource` is deliberately absent — Linux is descoped to this
    interface only (D17). On Linux the live branch therefore raises, and the
    message says what to do instead rather than leaving a bare ImportError.
    """
    if wav is not None:
        return WavFileSource(wav, realtime=realtime)

    if sys.platform == "win32":
        from .wasapi import WasapiLoopbackSource  # lazy (D22)

        return WasapiLoopbackSource(device_index=device_index)

    raise AudioDeviceUnavailable(
        f"no live capture source on {sys.platform!r}. Windows is the target "
        f"platform (D16) and Linux is descoped to the AudioSource interface "
        f"only (D17); the PipeWire monitor path was verified in D2 but is not "
        f"built. Pass a WAV file to run the pipeline here (D22)."
    )
