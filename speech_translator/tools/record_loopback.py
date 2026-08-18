"""``python -m speech_translator.tools.record_loopback``.

Captures system audio to a WAV. Two jobs:

1. **The benchmark's input.** `docs/BENCHMARK.md` step 2 sources its 10+ minutes
   of sample audio through this command rather than a curated file, so Level 3
   measures the real capture path. That puts this tool on the critical path.
2. **The proof that the format layer works.** What lands on disk is the exact
   byte stream `AudioSource.frames()` yields — 16 kHz mono int16 — so if the
   file plays back at the right pitch with no clicks at chunk boundaries, the
   decode → downmix → streaming resample → framing chain is correct. Writing the
   device-native stream instead would have made the file a recording *of the
   device* and proved nothing about the code (D47).

`--input-wav` swaps `WavFileSource` in for the live source. That is not a
convenience: it means the metering, the incremental writing, the Ctrl-C path and
the duration report all run and are tested on the Linux dev box, leaving device
open as the only part that genuinely needs the demo machine (D48).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config as cfg
from ..audio.select import open_source
from ..audio.source import AudioDeviceLost, AudioSourceError
from ..logging_setup import force_utf8

_PROGRESS_INTERVAL_S = 1.0


def _dbfs(peak: int) -> float:
    return -math.inf if peak <= 0 else 20.0 * math.log10(peak / 32767.0)


def _meter(peak: int, width: int = 20) -> str:
    db = _dbfs(peak)
    filled = 0 if db == -math.inf else max(0, min(width, int(round((db + 60.0) / 60.0 * width))))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _default_out(c: cfg.Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return c.data_dir / "recordings" / f"loopback-{stamp}.wav"


def record(
    out_path: Path,
    seconds: float,
    device_index: int | None = None,
    input_wav: str | None = None,
    realtime: bool = False,
    progress: bool = True,
) -> dict:
    """Capture to `out_path`; return a summary. Always leaves a valid WAV."""
    source = open_source(wav=input_wav, device_index=device_index, realtime=realtime)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    peak = 0
    window_peak = 0
    interrupted = False
    lost: str | None = None
    started = time.perf_counter()
    next_report = started + _PROGRESS_INTERVAL_S

    try:
        wav = wave.open(str(out_path), "wb")
        wav.setnchannels(cfg.CHANNELS)
        wav.setsampwidth(cfg.SAMPLE_WIDTH_BYTES)
        wav.setframerate(cfg.SAMPLE_RATE)
    except OSError:
        source.close()  # do not leak the device because the path was unwritable
        raise

    try:
        if progress:
            print(f"recording -> {out_path}", file=sys.stderr)
            print(f"source: {source.describe()}", file=sys.stderr)  # type: ignore[attr-defined]
        for frame in source.frames():
            # Written frame by frame, so a Ctrl-C at minute nine leaves nine
            # minutes of usable audio rather than a truncated header.
            wav.writeframes(frame)
            frames += 1

            block_peak = int(np.abs(np.frombuffer(frame, dtype=np.int16).astype(np.int32)).max())
            peak = max(peak, block_peak)
            window_peak = max(window_peak, block_peak)

            captured_s = frames * cfg.FRAME_MS / 1000.0
            if captured_s >= seconds:
                break

            now = time.perf_counter()
            if progress and now >= next_report:
                # A silent recording is the classic ten-minute waste; the meter
                # makes it obvious in the first few seconds.
                print(
                    f"\r  {captured_s:7.1f} / {seconds:.0f} s  {frames:>7} frames  "
                    f"{_meter(window_peak)} peak {_dbfs(window_peak):6.1f} dBFS ",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
                window_peak = 0
                next_report = now + _PROGRESS_INTERVAL_S
    except KeyboardInterrupt:
        interrupted = True
    except AudioDeviceLost as exc:
        lost = str(exc)
    finally:
        source.close()
        wav.close()
        if progress:
            print("", file=sys.stderr)

    elapsed = time.perf_counter() - started
    return {
        "path": str(out_path),
        "frames": frames,
        "captured_s": frames * cfg.FRAME_MS / 1000.0,
        "wall_clock_s": elapsed,
        "peak_dbfs": _dbfs(peak),
        "silent": peak == 0,
        "interrupted": interrupted,
        "device_lost": lost,
        "source": source.describe(),  # type: ignore[attr-defined]
        "live": input_wav is None,
    }


def render(summary: dict) -> str:
    lines = [
        f"wrote {summary['path']}",
        f"  {summary['captured_s']:.2f} s · {summary['frames']} frames × "
        f"{cfg.FRAME_BYTES} bytes · {cfg.SAMPLE_RATE} Hz mono int16",
        f"  source: {summary['source']}",
        f"  peak: {summary['peak_dbfs']:.1f} dBFS",
    ]
    if summary["live"]:
        # Drift only means something when the clock and the device are meant to
        # agree — replaying a file as fast as possible is not a timing claim.
        drift = summary["captured_s"] - summary["wall_clock_s"]
        lines.append(
            f"  wall clock: {summary['wall_clock_s']:.2f} s · drift {drift:+.2f} s"
        )
    if summary["silent"]:
        lines.append(
            "  WARNING: every sample is zero. Nothing was playing, or the wrong "
            "endpoint was captured — check `list_devices`."
        )
    if summary["interrupted"]:
        lines.append("  stopped by Ctrl-C; the file above is complete and valid.")
    if summary["device_lost"]:
        lines.append(f"  DEVICE LOST: {summary['device_lost']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    c = cfg.load_config()
    parser = argparse.ArgumentParser(
        prog="python -m speech_translator.tools.record_loopback",
        description=(
            "Record system audio to a 16 kHz mono int16 WAV — the pipeline's own "
            "frame format, and the input docs/BENCHMARK.md consumes."
        ),
    )
    parser.add_argument(
        "-t", "--seconds", type=float, default=600.0,
        help="how long to capture (default: 600, the 10 minutes BENCHMARK.md asks for)",
    )
    parser.add_argument("-o", "--out", type=Path, default=None, help="output WAV path")
    parser.add_argument("-d", "--device", type=int, default=None, help="loopback device index")
    parser.add_argument(
        "--input-wav", type=str, default=None,
        help="read a WAV instead of the device — the Linux smoke path (D22, D48)",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="with --input-wav, pace at wall-clock speed instead of as fast as possible",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress meter")
    args = parser.parse_args(argv)

    out = args.out or _default_out(c)
    try:
        summary = record(
            out_path=out,
            seconds=args.seconds,
            device_index=args.device,
            input_wav=args.input_wav,
            realtime=args.realtime,
            progress=not args.quiet,
        )
    except AudioSourceError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print(render(summary))
    return 1 if summary["silent"] or summary["device_lost"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
