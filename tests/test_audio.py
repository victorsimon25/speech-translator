"""Level 1 acceptance, as tests.

Runs on Linux with no audio hardware (D22). What cannot be tested here — opening
a real WASAPI endpoint, and whether ten minutes of it plays back cleanly — is
carried forward in `PLAN.md` rather than ticked off.

The test this file exists for is `test_streaming_resampler_has_no_chunk_seams`
and its control. Everything else is arithmetic; that one is the failure mode
`PLAN.md` calls this level's most likely way to ship broken.
"""

from __future__ import annotations

import struct
import sys
import time
import wave

import numpy as np
import pytest

from speech_translator import config as cfg
from speech_translator.audio import (
    AudioSource,
    FrameFormatter,
    UnsupportedAudioFormat,
    WasapiLoopbackSource,
    WavFileSource,
    open_source,
)
from speech_translator.audio.source import AudioDeviceUnavailable

TONE_HZ = 440.0
TONE_AMP = 20_000


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def tone(rate: int, seconds: float, hz: float = TONE_HZ, amp: int = TONE_AMP) -> np.ndarray:
    n = int(rate * seconds)
    return (np.sin(2 * np.pi * hz * np.arange(n) / rate) * amp).astype(np.int16)


def max_step(x: np.ndarray) -> int:
    """Largest sample-to-sample jump — how a click shows up numerically."""
    return int(np.abs(np.diff(x.astype(np.int32))).max())


def tone_step_bound(hz: float = TONE_HZ, amp: int = TONE_AMP, rate: int = cfg.SAMPLE_RATE) -> float:
    """Steepest slope a clean sine of these parameters can have, plus slack.

    A resampled sine cannot legitimately jump further than its own derivative
    allows; anything above this is a seam, not signal.
    """
    return amp * 2 * np.pi * hz / rate * 1.05


def write_wav(path, data: np.ndarray, rate: int, channels: int, sampwidth: int = 2) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def write_float32_wav(path, rate: int = 48_000, channels: int = 2, n: int = 1000) -> None:
    """A WAV the stdlib `wave` module refuses (format 3), written by hand."""
    data = np.zeros((n, channels), dtype=np.float32).tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, channels, rate, rate * channels * 4, channels * 4, 32))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def push_ragged(fmt: FrameFormatter, data: np.ndarray, seed: int = 0) -> list[bytes]:
    """Feed a signal in unpredictable chunk sizes, as a real device does."""
    rng = np.random.default_rng(seed)
    raw = data.tobytes()
    frames: list[bytes] = []
    i = 0
    while i < len(raw):
        n = int(rng.integers(1, 5000))
        frames += fmt.push(raw[i : i + n])
        i += n
    return frames + fmt.flush()


def frames_to_int16(frames: list[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(frames), dtype=np.int16)


# --------------------------------------------------------------------------
# Frame contract (INTERFACES.md §1)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rate", "channels"),
    [(16_000, 1), (48_000, 2), (44_100, 2), (48_000, 1), (96_000, 2)],
)
def test_every_frame_is_exactly_one_frame(rate, channels):
    """1024 bytes, always. Silero accepts nothing else (D23)."""
    sig = tone(rate, 1.3)
    data = np.repeat(sig[:, None], channels, axis=1) if channels > 1 else sig
    frames = push_ragged(FrameFormatter(rate, channels, "int16"), data)
    assert frames
    assert {len(f) for f in frames} == {cfg.FRAME_BYTES}


@pytest.mark.parametrize(("rate", "channels"), [(16_000, 1), (48_000, 2), (44_100, 2)])
def test_sample_count_matches_source_duration_within_one_frame(rate, channels):
    seconds = 2.5
    sig = tone(rate, seconds)
    data = np.repeat(sig[:, None], channels, axis=1) if channels > 1 else sig
    frames = push_ragged(FrameFormatter(rate, channels, "int16"), data)

    produced = len(frames) * cfg.FRAME_SAMPLES
    expected = seconds * cfg.SAMPLE_RATE
    assert abs(produced - expected) <= cfg.FRAME_SAMPLES


def test_16k_mono_and_48k_stereo_produce_the_same_shape():
    """Same duration in, same number of frames out — whatever the device is."""
    seconds = 3.0
    mono = tone(cfg.SAMPLE_RATE, seconds)
    stereo = np.repeat(tone(48_000, seconds)[:, None], 2, axis=1)

    a = push_ragged(FrameFormatter(cfg.SAMPLE_RATE, 1, "int16"), mono, seed=1)
    b = push_ragged(FrameFormatter(48_000, 2, "int16"), stereo, seed=2)

    assert len(a) == len(b)
    assert {len(f) for f in a} == {len(f) for f in b} == {cfg.FRAME_BYTES}


# --------------------------------------------------------------------------
# Bit-exactness — the 16 kHz mono path must not round-trip through float
# --------------------------------------------------------------------------


def test_16k_mono_int16_is_passthrough_and_bit_exact():
    sig = tone(cfg.SAMPLE_RATE, 1.7)
    fmt = FrameFormatter(cfg.SAMPLE_RATE, 1, "int16")
    assert fmt.passthrough is True

    frames = push_ragged(fmt, sig)
    out = b"".join(frames)
    src = sig.tobytes()
    assert out[: len(src)] == src           # not "close to" — identical bytes
    assert set(out[len(src) :]) <= {0}      # only the zero pad completing a frame


def test_passthrough_never_builds_a_resampler(monkeypatch):
    import soxr

    calls = []
    monkeypatch.setattr(soxr, "ResampleStream", lambda *a, **k: calls.append(a))
    FrameFormatter(cfg.SAMPLE_RATE, 1, "int16")
    assert calls == []


# --------------------------------------------------------------------------
# The seam test, and the control that proves it works
# --------------------------------------------------------------------------


def test_streaming_resampler_has_no_chunk_seams():
    """One ResampleStream, so filter state carries across chunks (D24, D46).

    Two independent assertions: no sample-to-sample jump steeper than the tone
    itself can be, and agreement with a single-shot resample of the whole
    signal. A seam breaks both.
    """
    src = tone(48_000, 3.0)
    fmt = FrameFormatter(48_000, 1, "int16")
    y = frames_to_int16(push_ragged(fmt, src, seed=7))

    assert max_step(y) <= tone_step_bound()

    import soxr

    reference = soxr.resample(src, 48_000, cfg.SAMPLE_RATE, quality="HQ")
    n = min(len(reference), len(y))
    assert n > cfg.SAMPLE_RATE  # the comparison must actually cover something
    assert np.abs(y[:n].astype(np.int32) - reference[:n].astype(np.int32)).max() <= 4


def test_control_a_per_chunk_resampler_does_produce_seams():
    """The bug the test above guards against, reproduced deliberately.

    Without this, `test_streaming_resampler_has_no_chunk_seams` could pass for
    the wrong reason — a bound loose enough to accept anything. This asserts the
    bound is tight enough to reject the actual failure. Measured on the dev box:
    3453 streaming vs 9087 per-chunk, against a bound of ~3505.
    """
    import soxr

    src = tone(48_000, 3.0)
    rng = np.random.default_rng(7)
    out = []
    i = 0
    while i < len(src):
        n = int(rng.integers(1, 5000))
        chunk = src[i : i + n]
        i += n
        # The wrong thing: a fresh resampler per chunk, discarding the delay
        # line each time.
        out.append(soxr.ResampleStream(48_000, cfg.SAMPLE_RATE, 1, dtype="int16").resample_chunk(chunk, last=True))
    y = np.concatenate(out)

    assert max_step(y) > tone_step_bound()


def test_resampler_is_constructed_exactly_once(monkeypatch):
    """Structural guard: the seam test proves the output, this proves the cause."""
    import soxr

    real = soxr.ResampleStream
    built = []

    def spy(*args, **kwargs):
        built.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(soxr, "ResampleStream", spy)

    fmt = FrameFormatter(48_000, 2, "int16")
    stereo = np.repeat(tone(48_000, 2.0)[:, None], 2, axis=1)
    push_ragged(fmt, stereo)

    assert len(built) == 1


# --------------------------------------------------------------------------
# Downmix and sample-format conversion
# --------------------------------------------------------------------------


def test_downmix_cancels_antiphase_and_preserves_identical_channels():
    sig = tone(cfg.SAMPLE_RATE, 0.5)

    antiphase = np.stack([sig, -sig], axis=1)
    y = frames_to_int16(push_ragged(FrameFormatter(cfg.SAMPLE_RATE, 2, "int16"), antiphase))
    assert int(np.abs(y.astype(np.int32)).max()) == 0

    duplicated = np.repeat(sig[:, None], 2, axis=1)
    y2 = frames_to_int16(push_ragged(FrameFormatter(cfg.SAMPLE_RATE, 2, "int16"), duplicated))
    assert np.array_equal(y2[: len(sig)], sig)


def test_downmix_of_two_full_scale_channels_does_not_wrap():
    """int16 + int16 overflows int16; a wrapped sample is a loud click."""
    loud = np.full((512, 2), 32_000, dtype=np.int16)
    y = frames_to_int16(push_ragged(FrameFormatter(cfg.SAMPLE_RATE, 2, "int16"), loud))
    assert y.min() >= 0 and y.max() == 32_000


def test_float32_input_saturates_rather_than_wrapping():
    """WASAPI may hand back float32, and a mixed desktop can exceed unity."""
    pattern = [2.0, -2.0, 1.0, -1.0, 0.5, 0.0]
    hot = np.array(pattern * 200, dtype=np.float32)
    fmt = FrameFormatter(cfg.SAMPLE_RATE, 1, "float32")
    assert fmt.passthrough is False

    y = frames_to_int16(push_ragged(fmt, hot))[: len(hot)]
    # 16 kHz in, 16 kHz out: no resampling, so samples map one to one and the
    # whole conversion can be checked element-wise rather than statistically.
    expected = np.array([32_767, -32_767, 32_767, -32_767, 16_384, 0] * 200, dtype=np.int16)
    assert np.array_equal(y, expected)


def test_unsupported_sample_format_is_named():
    with pytest.raises(UnsupportedAudioFormat, match="complex"):
        FrameFormatter(48_000, 1, "complex64")


def test_push_after_flush_is_refused():
    from speech_translator.audio.format import AudioFormatterClosed

    fmt = FrameFormatter(48_000, 1, "int16")
    fmt.push(tone(48_000, 0.1).tobytes())
    fmt.flush()
    with pytest.raises(AudioFormatterClosed):
        fmt.push(b"\x00" * 1024)


def test_a_sheared_sample_is_carried_not_dropped():
    """Odd-sized reads must not shear a sample in half and lose a byte."""
    sig = tone(48_000, 1.0)
    raw = sig.tobytes()
    fmt = FrameFormatter(48_000, 1, "int16")
    for i in range(0, len(raw), 333):  # never a whole number of samples
        fmt.push(raw[i : i + 333])
    fmt.flush()
    assert fmt.src_samples_in == len(sig)


# --------------------------------------------------------------------------
# WavFileSource
# --------------------------------------------------------------------------


def test_wav_source_frame_count_matches_duration(tmp_path):
    path = tmp_path / "s.wav"
    write_wav(path, np.repeat(tone(48_000, 2.0)[:, None], 2, axis=1), 48_000, 2)

    with WavFileSource(path) as src:
        frames = list(src.frames())

    assert {len(f) for f in frames} == {cfg.FRAME_BYTES}
    assert abs(len(frames) * cfg.FRAME_SAMPLES - 2.0 * cfg.SAMPLE_RATE) <= cfg.FRAME_SAMPLES


def test_wav_source_round_trips_our_own_recording_bit_exactly(tmp_path):
    """A record_loopback capture replayed must be identical, not resampled twice."""
    path = tmp_path / "capture.wav"
    sig = tone(cfg.SAMPLE_RATE, 1.5)
    write_wav(path, sig, cfg.SAMPLE_RATE, 1)

    with WavFileSource(path) as src:
        assert src.formatter.passthrough is True
        out = b"".join(src.frames())

    assert out[: sig.nbytes] == sig.tobytes()


def test_wav_source_24bit_is_decoded(tmp_path):
    path = tmp_path / "p24.wav"
    sig = tone(cfg.SAMPLE_RATE, 0.5).astype(np.int32) << 8
    raw = np.frombuffer(sig.tobytes(), dtype=np.uint8).reshape(-1, 4)[:, 1:].tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(cfg.SAMPLE_RATE)
        w.writeframes(raw)

    with WavFileSource(path) as src:
        y = frames_to_int16(list(src.frames()))
    assert np.array_equal(y[: len(sig)], (sig >> 16).astype(np.int16))


def test_wav_source_realtime_paces_at_wall_clock(tmp_path):
    path = tmp_path / "half.wav"
    write_wav(path, tone(cfg.SAMPLE_RATE, 0.5), cfg.SAMPLE_RATE, 1)

    t0 = time.perf_counter()
    with WavFileSource(path, realtime=True) as src:
        n = sum(1 for _ in src.frames())
    elapsed = time.perf_counter() - t0

    assert n == pytest.approx(0.5 * cfg.SAMPLE_RATE / cfg.FRAME_SAMPLES, abs=1)
    assert 0.35 < elapsed < 0.95  # paced, not instant; generous for a loaded CI box


def test_wav_source_close_releases_the_file(tmp_path):
    path = tmp_path / "s.wav"
    write_wav(path, tone(cfg.SAMPLE_RATE, 0.2), cfg.SAMPLE_RATE, 1)
    src = WavFileSource(path)
    handle = src._wav._file.file       # the real OS-level file object
    assert list(src.frames())          # it is live
    assert not handle.closed
    src.close()
    src.close()                        # idempotent
    assert handle.closed


def test_float32_wav_is_refused_with_a_useful_message(tmp_path):
    path = tmp_path / "f32.wav"
    write_float32_wav(path)
    with pytest.raises(UnsupportedAudioFormat) as exc:
        WavFileSource(path)
    assert "record_loopback" in str(exc.value)


# --------------------------------------------------------------------------
# The Protocol, and the Linux-importability rule (D22)
# --------------------------------------------------------------------------


def test_both_sources_satisfy_the_protocol(tmp_path):
    path = tmp_path / "s.wav"
    write_wav(path, tone(cfg.SAMPLE_RATE, 0.1), cfg.SAMPLE_RATE, 1)
    with WavFileSource(path) as src:
        assert isinstance(src, AudioSource)
    assert isinstance(WasapiLoopbackSource(), AudioSource)


def test_importing_the_audio_package_does_not_import_pyaudiowpatch():
    """D22: the dev box has no pyaudiowpatch, and must still import the package."""
    import subprocess

    code = (
        "import sys, speech_translator.audio as a;"
        "a.WasapiLoopbackSource();"
        "print('pyaudiowpatch' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, cwd=cfg.REPO_ROOT
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert proc.stdout.decode().strip() == "False"


@pytest.mark.skipif(sys.platform == "win32", reason="Linux dev-box expectation")
def test_wasapi_use_on_linux_fails_with_an_actionable_message():
    with pytest.raises(AudioDeviceUnavailable) as exc:
        next(WasapiLoopbackSource().frames())
    assert "pyaudiowpatch" in str(exc.value)
    assert "WavFileSource" in str(exc.value)


@pytest.mark.skipif(sys.platform == "win32", reason="Linux dev-box expectation")
def test_open_source_has_no_live_source_on_linux(tmp_path):
    """D17: PipeWireMonitorSource is not built, and says so."""
    with pytest.raises(AudioDeviceUnavailable, match="D17"):
        open_source()

    path = tmp_path / "s.wav"
    write_wav(path, tone(cfg.SAMPLE_RATE, 0.1), cfg.SAMPLE_RATE, 1)
    assert isinstance(open_source(wav=path, realtime=False), WavFileSource)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="Linux dev-box expectation")
def test_list_devices_exits_non_zero_on_linux(capsys):
    from speech_translator.tools import list_devices

    assert list_devices.main([]) == 2
    assert "Windows-only" in capsys.readouterr().err


def test_record_loopback_from_a_wav_end_to_end(tmp_path):
    """The Linux smoke path (D48): everything but device open actually runs."""
    from speech_translator.tools import record_loopback

    src_path = tmp_path / "in.wav"
    out_path = tmp_path / "out.wav"
    write_wav(src_path, np.repeat(tone(48_000, 2.0)[:, None], 2, axis=1), 48_000, 2)

    rc = record_loopback.main(
        ["--input-wav", str(src_path), "--out", str(out_path), "--quiet"]
    )
    assert rc == 0

    with wave.open(str(out_path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
            cfg.SAMPLE_RATE,
            cfg.CHANNELS,
            cfg.SAMPLE_WIDTH_BYTES,
        )
        assert abs(w.getnframes() - 2.0 * cfg.SAMPLE_RATE) <= cfg.FRAME_SAMPLES

    # And what it wrote is replayable through the bit-exact passthrough path.
    with WavFileSource(out_path) as replay:
        assert replay.formatter.passthrough is True
        assert len(list(replay.frames())) == w.getnframes() // cfg.FRAME_SAMPLES


def test_record_loopback_honours_the_seconds_limit(tmp_path):
    from speech_translator.tools import record_loopback

    src_path = tmp_path / "in.wav"
    out_path = tmp_path / "out.wav"
    write_wav(src_path, tone(cfg.SAMPLE_RATE, 3.0), cfg.SAMPLE_RATE, 1)

    summary = record_loopback.record(
        out_path=out_path, seconds=1.0, input_wav=str(src_path), progress=False
    )
    assert summary["captured_s"] == pytest.approx(1.0, abs=cfg.FRAME_MS / 1000.0)
    assert summary["silent"] is False


def test_record_loopback_flags_a_silent_capture(tmp_path):
    """Ten minutes of zeros is the classic wasted recording; it must be named."""
    from speech_translator.tools import record_loopback

    src_path = tmp_path / "quiet.wav"
    out_path = tmp_path / "out.wav"
    write_wav(src_path, np.zeros(cfg.SAMPLE_RATE, dtype=np.int16), cfg.SAMPLE_RATE, 1)

    summary = record_loopback.record(
        out_path=out_path, seconds=5.0, input_wav=str(src_path), progress=False
    )
    assert summary["silent"] is True
    assert "WARNING" in record_loopback.render(summary)
