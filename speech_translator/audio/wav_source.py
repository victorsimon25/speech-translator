"""`WavFileSource` — a WAV file behind the `AudioSource` contract.

Two jobs (D4, D22):

* **Deterministic tests.** `realtime=False` runs as fast as the CPU allows, so
  the whole format layer is provable on the Linux dev box with no audio
  hardware.
* **The Linux smoke path.** `realtime=True` paces frames at wall-clock speed, so
  the downstream pipeline sees the same timing it will see live and the dev loop
  is not push-pull-run for every change.

It is also how a `record_loopback` capture gets replayed: our own recordings are
16 kHz mono int16, which is the formatter's passthrough case, so a replay is
bit-identical to the capture rather than resampled twice.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import FRAME_MS
from .format import FrameFormatter
from .source import AudioSourceError, UnsupportedAudioFormat

#: How much to read per `wave.readframes` call. Arbitrary — the formatter
#: handles any chunk size, and the test suite deliberately uses ragged ones.
_READ_FRAMES = 4096


def _decode_24bit(raw: bytes, channels: int) -> np.ndarray:
    """24-bit little-endian PCM → int32 scaled to the full int32 range.

    `wave` reports these as `sampwidth == 3`; numpy has no 24-bit type, so the
    bytes are widened here and the formatter's int32 path takes it from there.
    """
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    # Little-endian: pack into the *top* three bytes of an int32 so the sign bit
    # lands in the right place and the value keeps full scale.
    packed = (b[:, 0].astype(np.uint32) << 8) | (b[:, 1].astype(np.uint32) << 16) | (
        b[:, 2].astype(np.uint32) << 24
    )
    return packed.astype(np.int32).reshape(-1, channels)


class WavFileSource:
    """Reads a WAV file and yields 1024-byte frames in the pipeline format."""

    def __init__(self, path: str | Path, realtime: bool = False) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self._closed = False

        if not self.path.is_file():
            raise AudioSourceError(f"no such WAV file: {self.path}")
        try:
            self._wav = wave.open(str(self.path), "rb")
        except wave.Error as exc:
            # The common one is a float32 WAV (`unknown format: 3`), which the
            # stdlib reader does not handle. Say so, rather than "wave.Error".
            raise UnsupportedAudioFormat(
                f"{self.path}: {exc} — WavFileSource reads integer PCM only "
                f"(8/16/24/32-bit). Re-encode to 16-bit PCM, or capture with "
                f"`python -m speech_translator.tools.record_loopback`, which "
                f"writes 16 kHz mono int16."
            ) from exc

        self.src_rate = self._wav.getframerate()
        self.src_channels = self._wav.getnchannels()
        self._sampwidth = self._wav.getsampwidth()
        self.duration_s = self._wav.getnframes() / self.src_rate if self.src_rate else 0.0

        dtype = {1: "uint8", 2: "int16", 3: "int32", 4: "int32"}.get(self._sampwidth)
        if dtype is None:
            self._wav.close()
            raise UnsupportedAudioFormat(
                f"{self.path}: {self._sampwidth * 8}-bit samples are not supported"
            )

        self.formatter = FrameFormatter(self.src_rate, self.src_channels, dtype)

    # -- AudioSource -------------------------------------------------------

    def frames(self) -> Iterator[bytes]:
        started = time.perf_counter()
        index = 0

        while not self._closed:
            raw = self._wav.readframes(_READ_FRAMES)
            if not raw:
                break
            chunk = _decode_24bit(raw, self.src_channels) if self._sampwidth == 3 else raw
            for frame in self.formatter.push(chunk):
                if self._closed:
                    return
                if self.realtime:
                    # Pace against a cumulative index, not sleep(0.032) per
                    # frame — the latter accumulates the scheduler's error and
                    # drifts away from wall clock over a ten-minute file.
                    due = started + (index * FRAME_MS) / 1000.0
                    delay = due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                index += 1
                yield frame

        if not self._closed:
            for frame in self.formatter.flush():
                yield frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wav.close()

    # -- convenience -------------------------------------------------------

    def __enter__(self) -> "WavFileSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def describe(self) -> str:
        return f"{self.path.name} · {self.duration_s:.2f} s · {self.formatter.describe()}"
