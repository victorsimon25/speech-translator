"""`WasapiLoopbackSource` — the live source (D3).

Captures what the *output* device is playing, not a microphone (D1). On Windows
that is WASAPI loopback, reached through `PyAudioWPatch`.

**The import is lazy.** `pyaudiowpatch` carries a `sys_platform == 'win32'`
marker and is not installed on the Linux dev box; if this module imported it at
module scope, `import speech_translator.audio` would fail on the machine the
code is written on and the dev loop would become push-pull-run for every typo
(D22). Everything here is importable everywhere; only `_open()` needs Windows.

**The device's format is read, never assumed.** `defaultSampleRate` is whatever
the user's output device is configured for — 48 000 is typical, 44 100 and
96 000 are not exotic. Hard-coding 48000 is on the D41 register. The rate and
channel count come off the device info and go straight into the `FrameFormatter`
(D24).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from ..config import FRAME_MS
from .format import FrameFormatter
from .source import AudioDeviceLost, AudioDeviceUnavailable

log = logging.getLogger(__name__)


class WasapiLoopbackSource:
    """WASAPI loopback capture, normalised to the pipeline frame format."""

    def __init__(self, device_index: int | None = None) -> None:
        self.device_index = device_index
        self._pa: Any = None
        self._stream: Any = None
        self._closed = False

        self.device_name: str = ""
        self.src_rate: int = 0
        self.src_channels: int = 0
        self.sample_format: str = ""
        self.formatter: FrameFormatter | None = None

    # -- device selection --------------------------------------------------

    @staticmethod
    def _import_pyaudio() -> Any:
        try:
            import pyaudiowpatch as pyaudio  # noqa: PLC0415 — lazy on purpose (D22)
        except ImportError as exc:
            raise AudioDeviceUnavailable(
                "pyaudiowpatch is not available. It is a Windows-only dependency "
                "(marked sys_platform == 'win32' in pyproject.toml) and WASAPI "
                "loopback exists only on Windows (D3, D17). On Linux use "
                "WavFileSource — that is the supported dev path (D22)."
            ) from exc
        return pyaudio

    @classmethod
    def loopback_devices(cls) -> list[dict]:
        """Every WASAPI loopback device, default output first if we can tell."""
        pyaudio = cls._import_pyaudio()
        pa = pyaudio.PyAudio()
        try:
            devices = list(pa.get_loopback_device_info_generator())
            try:
                wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
            except Exception:  # noqa: BLE001 — enumeration must not raise
                default_out = ""
            for d in devices:
                d["is_default_output"] = bool(default_out) and default_out in d["name"]
            devices.sort(key=lambda d: not d["is_default_output"])
            return devices
        finally:
            pa.terminate()

    def _select_device(self, pa: Any, pyaudio: Any) -> dict:
        if self.device_index is not None:
            return pa.get_device_info_by_index(self.device_index)

        devices = list(pa.get_loopback_device_info_generator())
        if not devices:
            raise AudioDeviceUnavailable(
                "no WASAPI loopback device found. Windows exposes one per output "
                "device; if there are none, no audio endpoint is active. Run "
                "`python -m speech_translator.tools.list_devices`."
            )
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
        except Exception:  # noqa: BLE001
            default_out = ""
        # Match the default output device, so we capture what the user hears
        # rather than an arbitrary endpoint.
        return next((d for d in devices if default_out and default_out in d["name"]), devices[0])

    # -- open / read -------------------------------------------------------

    def _open(self) -> None:
        pyaudio = self._import_pyaudio()
        self._pa = pyaudio.PyAudio()
        try:
            info = self._select_device(self._pa, pyaudio)
        except Exception:
            self._pa.terminate()
            self._pa = None
            raise

        self.device_name = str(info["name"])
        self.device_index = int(info["index"])
        # Never hard-coded (D41): the device says what it is.
        self.src_rate = int(info["defaultSampleRate"])
        self.src_channels = int(info["maxInputChannels"])
        self._chunk = max(256, int(self.src_rate * FRAME_MS / 1000))

        # Ask for int16 first: it is what the pipeline wants and skips a
        # conversion. WASAPI shared mode may insist on the mix format, which is
        # commonly float32 — so fall back rather than fail, and record which one
        # we got, because that is a fact about the demo machine worth knowing.
        last_exc: Exception | None = None
        for fmt_name, pa_fmt, np_dtype in (
            ("int16", pyaudio.paInt16, "int16"),
            ("float32", pyaudio.paFloat32, "float32"),
        ):
            try:
                self._stream = self._pa.open(
                    format=pa_fmt,
                    channels=self.src_channels,
                    rate=self.src_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self._chunk,
                )
            except Exception as exc:  # noqa: BLE001 — try the next format
                last_exc = exc
                continue
            self.sample_format = fmt_name
            self.formatter = FrameFormatter(self.src_rate, self.src_channels, np_dtype)
            log.info("WASAPI loopback: %s · %s", self.device_name, self.formatter.describe())
            return

        self._pa.terminate()
        self._pa = None
        raise AudioDeviceUnavailable(
            f"could not open {self.device_name} at {self.src_rate} Hz "
            f"{self.src_channels} ch in int16 or float32: {last_exc}"
        )

    def frames(self) -> Iterator[bytes]:
        if self._stream is None:
            self._open()
        assert self.formatter is not None

        while not self._closed:
            try:
                data = self._stream.read(self._chunk, exception_on_overflow=False)
            except OSError as exc:
                # The realistic cause is on the D41 register: the default output
                # device changed mid-session (headphones plugged in) and this
                # endpoint no longer exists. Surface it; Level 6 turns it into
                # the `error` state with a re-select action rather than a silent
                # stall.
                self.close()
                raise AudioDeviceLost(
                    f"loopback stream on {self.device_name!r} failed mid-session: {exc}. "
                    f"The default output device most likely changed."
                ) from exc
            for frame in self.formatter.push(data):
                if self._closed:
                    return
                yield frame

        for frame in self.formatter.flush():
            yield frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # noqa: BLE001 — closing must not raise
                pass
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def __enter__(self) -> "WasapiLoopbackSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def describe(self) -> str:
        if self.formatter is None:
            return "WASAPI loopback (not open)"
        return f"{self.device_name} · {self.sample_format} · {self.formatter.describe()}"
