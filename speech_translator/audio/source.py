"""The `AudioSource` contract and its errors.

`INTERFACES.md` §1 defines the Protocol; this file is that definition in code
and adds nothing to it. Every implementation yields **16 kHz mono signed 16-bit
little-endian PCM in exactly 512-sample (1024-byte) frames** — normalisation is
the source's job, not a downstream stage's (D24), so nothing after this layer
contains a sample-rate branch.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class AudioSource(Protocol):
    """Yields raw PCM frames in the fixed pipeline format."""

    def frames(self) -> Iterator[bytes]:
        """Yield 1024-byte frames until the stream ends or `close()` is called."""
        ...

    def close(self) -> None:
        """Release the device or file handle. Idempotent."""
        ...


class AudioSourceError(RuntimeError):
    """Base class for everything this layer raises."""


class UnsupportedAudioFormat(AudioSourceError):
    """The input exists but is in a format this source cannot decode."""


class AudioDeviceUnavailable(AudioSourceError):
    """The device could not be opened — wrong platform, or nothing to capture."""


class AudioDeviceLost(AudioSourceError):
    """The stream died mid-session.

    The realistic cause is the D41 register's "default output device changes
    mid-session (headphones)". Level 6 maps this to the `error` state with a
    re-select action; the one thing it must not do is die silently.
    """
