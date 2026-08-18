"""decode → downmix → streaming resample → exact 512-sample frames.

This is the whole of D24. WASAPI loopback hands back the *output device's*
native format — usually 48 kHz stereo, sometimes float32 — and Whisper and
Silero both want 16 kHz mono int16 in 512-sample frames (D23). The conversion
happens here, inside the source, so the `AudioSource` contract in
`INTERFACES.md` §1 is literally true and no downstream stage ever branches on a
sample rate.

**One `soxr.ResampleStream` for the life of the stream.** The resampler is
stateful: it is a polyphase filter with a delay line, and that line has to carry
across chunk boundaries. Rebuilding it per chunk — the obvious-looking
`soxr.resample(chunk, 48000, 16000)` — discards the delay line each time and
welds together fragments that do not line up, which is audible as a click at
every boundary and is the single most likely way this level ships broken. See
D46 for the measurement, and `tests/test_audio.py` for the guard, which includes
a control proving the test detects the bug it exists for.

The 16 kHz mono int16 case is not converted at all: no downmix, no resampler, no
float round-trip. `passthrough` is True and the bytes reach the framer
untouched, so replaying our own recordings is bit-exact by construction rather
than by rounding luck.
"""

from __future__ import annotations

import numpy as np

from ..config import CHANNELS, FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE
from .source import UnsupportedAudioFormat

#: Sample formats soxr's streaming resampler accepts. Everything else is
#: widened into one of these before it reaches the resampler.
_SOXR_DTYPES = frozenset({np.dtype("int16"), np.dtype("int32"), np.dtype("float32"), np.dtype("float64")})


def _work_dtype(src: np.dtype) -> np.dtype:
    """The dtype the chain runs in, given what the device or file produces."""
    if src == np.uint8:
        return np.dtype("int16")  # 8-bit WAV is unsigned; widen on decode
    if src == np.float64:
        return np.dtype("float32")  # no reason to resample at double precision
    if src in _SOXR_DTYPES:
        return src
    raise UnsupportedAudioFormat(f"unsupported sample format: {src}")


class FrameFormatter:
    """Turns arbitrary-sized chunks of device audio into exact 1024-byte frames.

    Owned by an `AudioSource`, never used downstream. Not thread-safe: one
    instance belongs to one stream, and one stream is read by one thread.
    """

    def __init__(
        self,
        src_rate: int,
        src_channels: int,
        src_dtype: str | np.dtype = "int16",
        quality: str = "HQ",
    ) -> None:
        if src_rate <= 0:
            raise UnsupportedAudioFormat(f"non-positive sample rate: {src_rate}")
        if src_channels < 1:
            raise UnsupportedAudioFormat(f"non-positive channel count: {src_channels}")

        self.src_rate = int(src_rate)
        self.src_channels = int(src_channels)
        self.src_dtype = np.dtype(src_dtype)
        self.work_dtype = _work_dtype(self.src_dtype)

        #: Nothing to do at all: right rate, right layout, right sample format.
        self.passthrough = (
            self.src_rate == SAMPLE_RATE
            and self.src_channels == CHANNELS
            and self.work_dtype == np.dtype("int16")
            and self.src_dtype == np.dtype("int16")
        )

        self._resampler = None
        if self.src_rate != SAMPLE_RATE:
            import soxr  # local: keeps the module cheap to import

            # Built once, here. See the module docstring and D46 — this object
            # holding its filter state across chunks is the point.
            self._resampler = soxr.ResampleStream(
                self.src_rate,
                SAMPLE_RATE,
                CHANNELS,
                dtype=self.work_dtype.name,
                quality=quality,
            )

        self._partial = b""          # bytes of an incomplete sample-frame
        self._out = bytearray()      # 16 kHz mono int16, not yet framed
        self._flushed = False

        self.src_samples_in = 0      # per-channel input samples seen
        self.samples_out = 0         # 16 kHz samples produced, before framing
        self.frames_emitted = 0
        self.pad_samples = 0         # zeros added to complete the final frame

    # -- the chain ---------------------------------------------------------

    def _decode(self, data: bytes) -> np.ndarray:
        """bytes → (n, channels) array, carrying any sheared trailing sample."""
        buf = self._partial + data if self._partial else data
        stride = self.src_dtype.itemsize * self.src_channels
        usable = len(buf) - (len(buf) % stride)
        self._partial = buf[usable:]
        if usable == 0:
            return np.empty((0, self.src_channels), dtype=self.src_dtype)
        return np.frombuffer(buf, dtype=self.src_dtype, count=usable // self.src_dtype.itemsize).reshape(
            -1, self.src_channels
        )

    def _to_work(self, x: np.ndarray) -> np.ndarray:
        if self.src_dtype == np.uint8:
            return (x.astype(np.int16) - 128) << 8
        if self.src_dtype != self.work_dtype:
            return x.astype(self.work_dtype)
        return x

    def _downmix(self, x: np.ndarray) -> np.ndarray:
        """(n, ch) → (n,). Before the resampler, so soxr does one channel."""
        if x.shape[1] == 1:
            return x[:, 0]
        if self.work_dtype.kind == "f":
            return x.mean(axis=1, dtype=self.work_dtype)
        # Sum in a wider integer type first: two full-scale int16 channels
        # overflow int16, and a wrapped sample is a loud click.
        wide = np.int64 if self.work_dtype == np.int32 else np.int32
        return (x.astype(wide).sum(axis=1) // x.shape[1]).astype(self.work_dtype)

    def _resample(self, mono: np.ndarray, last: bool = False) -> np.ndarray:
        if self._resampler is None:
            return mono
        return self._resampler.resample_chunk(mono, last=last)

    def _to_int16(self, x: np.ndarray) -> np.ndarray:
        if self.work_dtype == np.int16:
            return x
        if self.work_dtype == np.int32:
            return (x >> 16).astype(np.int16)
        # Float: clip *before* scaling. Loopback audio can exceed unity after a
        # mix, and a wrapped sample is far worse than a clipped one.
        return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)

    # -- public ------------------------------------------------------------

    def push(self, data: bytes | np.ndarray) -> list[bytes]:
        """Feed a chunk; get back zero or more complete 1024-byte frames."""
        if self._flushed:
            raise AudioFormatterClosed("push() after flush() — build a new formatter")

        if isinstance(data, np.ndarray):
            x = data if data.ndim == 2 else data.reshape(-1, self.src_channels)
            x = x.astype(self.src_dtype, copy=False)
        else:
            x = self._decode(bytes(data))
        self.src_samples_in += x.shape[0]

        if self.passthrough:
            self._out += x[:, 0].tobytes()
            self.samples_out += x.shape[0]
        elif x.shape[0]:
            y = self._to_int16(self._resample(self._downmix(self._to_work(x))))
            self._out += y.tobytes()
            self.samples_out += y.shape[0]

        return self._drain()

    def flush(self) -> list[bytes]:
        """Drain the resampler's delay line and zero-pad the final frame.

        The pad is why "within one frame" is the right tolerance on total sample
        count: the VAD needs exactly 512 samples, so a partial tail is completed
        rather than dropped.
        """
        if self._flushed:
            return []
        self._flushed = True

        if self._resampler is not None:
            tail = self._resample(np.zeros(0, dtype=self.work_dtype), last=True)
            if tail.shape[0]:
                y = self._to_int16(tail)
                self._out += y.tobytes()
                self.samples_out += y.shape[0]

        frames = self._drain()
        if self._out:
            missing = FRAME_BYTES - len(self._out)
            self._out += b"\x00" * missing
            self.pad_samples = missing // 2
            frames += self._drain()
        return frames

    def _drain(self) -> list[bytes]:
        out: list[bytes] = []
        while len(self._out) >= FRAME_BYTES:
            out.append(bytes(self._out[:FRAME_BYTES]))
            del self._out[:FRAME_BYTES]
        self.frames_emitted += len(out)
        return out

    def describe(self) -> str:
        """One line for a log or a tool banner — what the device actually gave us."""
        if self.passthrough:
            return f"{self.src_rate} Hz mono int16 → passthrough (bit-exact, no resampler)"
        return (
            f"{self.src_rate} Hz {self.src_channels} ch {self.src_dtype.name} → "
            f"{SAMPLE_RATE} Hz mono int16 · "
            f"{'streaming resample' if self._resampler is not None else 'no resample'} · "
            f"{FRAME_SAMPLES}-sample frames"
        )


class AudioFormatterClosed(RuntimeError):
    """`push()` was called after `flush()`; the delay line is already drained."""
