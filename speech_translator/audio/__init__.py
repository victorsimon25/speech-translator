"""Stage 1 of the pipeline: audio in, 16 kHz mono int16 frames out.

Importing this package must succeed on Linux (D22), so `WasapiLoopbackSource`
is re-exported from a module that keeps its own `pyaudiowpatch` import inside
`_open()`. Importing the name is safe anywhere; only calling it needs Windows.
"""

from __future__ import annotations

from .format import FrameFormatter
from .select import open_source
from .source import (
    AudioDeviceLost,
    AudioDeviceUnavailable,
    AudioSource,
    AudioSourceError,
    UnsupportedAudioFormat,
)
from .wasapi import WasapiLoopbackSource
from .wav_source import WavFileSource

__all__ = [
    "AudioDeviceLost",
    "AudioDeviceUnavailable",
    "AudioSource",
    "AudioSourceError",
    "FrameFormatter",
    "UnsupportedAudioFormat",
    "WasapiLoopbackSource",
    "WavFileSource",
    "open_source",
]
