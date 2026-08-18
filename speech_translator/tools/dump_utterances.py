"""``python -m speech_translator.tools.dump_utterances``.

Runs audio through the Segmenter and shows what came out. This is Level 2's
inspection instrument, and it exists mainly for one acceptance criterion that no
unit test can close: *"on a captured meeting WAV, boundaries land at real pauses
on inspection"*. Inspection needs something to inspect.

Two modes, through the same `open_source()` seam `record_loopback` uses (D48):
`--input-wav` reads a file anywhere, and with no argument it opens the live
loopback device, which only exists on Windows. So the tool is fully exercised on
the dev box and needs no new code on the demo machine.

`--write-wav DIR` writes each utterance as its own WAV. That is the actual
answer to "did the boundaries land at real pauses" — you play them. A table of
millisecond offsets tells you the state machine is self-consistent, not that it
cut where a person would have.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .. import config as cfg
from ..audio.select import open_source
from ..audio.source import AudioDeviceLost, AudioSourceError
from ..logging_setup import force_utf8
from ..segment import Segmenter, SileroVad, Utterance


def _dbfs(peak: int) -> float:
    return -math.inf if peak <= 0 else 20.0 * math.log10(peak / 32767.0)


def _row(u: Utterance, peak_dbfs: float) -> str:
    return (
        f"{u.id}  {u.start_ms:8d} {u.end_ms:8d} {u.duration_ms:7d}  "
        f"{u.closed_by:<10} {u.preceding_silence_ms:7d}  {peak_dbfs:7.1f}"
    )


HEADER = (
    f"{'id':<7} {'start_ms':>8} {'end_ms':>8} {'dur_ms':>7}  "
    f"{'closed_by':<10} {'gap_ms':>7}  {'peak':>7}"
)


def _write_utterance(directory: Path, u: Utterance) -> Path:
    path = directory / f"{u.id}_{u.start_ms}-{u.end_ms}ms_{u.closed_by}.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(cfg.CHANNELS)
        wav.setsampwidth(cfg.SAMPLE_WIDTH_BYTES)
        wav.setframerate(cfg.SAMPLE_RATE)
        wav.writeframes(u.pcm)
    return path


def dump(
    input_wav: str | None = None,
    device_index: int | None = None,
    realtime: bool = False,
    max_utterance_ms: int | None = None,
    write_wav: Path | None = None,
    as_json: bool = False,
    config: cfg.Config | None = None,
) -> dict:
    """Segment a source; print as we go; return a summary."""
    c = config or cfg.load_config()
    source = open_source(wav=input_wav, device_index=device_index, realtime=realtime)
    segmenter = Segmenter(c, SileroVad())
    if max_utterance_ms is not None:
        # Exactly the call Level 6's backpressure controller will make (D28).
        segmenter.set_max_utterance_ms(max_utterance_ms)

    if not as_json:
        print(f"source:    {source.describe()}", file=sys.stderr)  # type: ignore[attr-defined]
        print(f"vad:       {segmenter.vad.describe()}", file=sys.stderr)  # type: ignore[attr-defined]
        print(
            f"thresholds: open {c.vad_speech_threshold} / release "
            f"{c.vad_release_threshold} · silence {c.silence_threshold_ms} ms · "
            f"max {segmenter.effective_max_utterance_ms} ms · min "
            f"{c.min_utterance_ms} ms",
            file=sys.stderr,
        )
        print(HEADER, file=sys.stderr)

    rows: list[dict] = []
    if write_wav is not None:
        write_wav.mkdir(parents=True, exist_ok=True)

    lost: str | None = None
    interrupted = False
    vad_seconds = 0.0
    started = time.perf_counter()

    # Time the VAD alone, not the file reading around it, because the number
    # this feeds is the D35 per-frame cost.
    original_probability = segmenter.vad.probability

    def timed_probability(frame: bytes) -> float:
        nonlocal vad_seconds
        t0 = time.perf_counter()
        try:
            return original_probability(frame)
        finally:
            vad_seconds += time.perf_counter() - t0

    segmenter.vad.probability = timed_probability  # type: ignore[method-assign]

    try:
        for u in segmenter.utterances(source.frames()):
            peak = int(np.abs(np.frombuffer(u.pcm, dtype=np.int16).astype(np.int32)).max())
            record = {
                "id": u.id,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
                "duration_ms": u.duration_ms,
                "closed_by": u.closed_by,
                "preceding_silence_ms": u.preceding_silence_ms,
                "bytes": len(u.pcm),
                "peak_dbfs": round(_dbfs(peak), 1),
            }
            if write_wav is not None:
                record["wav"] = str(_write_utterance(write_wav, u))
            rows.append(record)
            print(json.dumps(record) if as_json else _row(u, _dbfs(peak)), flush=True)
    except KeyboardInterrupt:
        interrupted = True
    except AudioDeviceLost as exc:
        lost = str(exc)
    finally:
        source.close()

    audio_ms = segmenter.frames_pushed * cfg.FRAME_MS
    speech_ms = segmenter.speech_frames * cfg.FRAME_MS
    return {
        "utterances": rows,
        "frames": segmenter.frames_pushed,
        "audio_ms": audio_ms,
        "speech_ms": speech_ms,
        "speech_ratio": (speech_ms / audio_ms) if audio_ms else 0.0,
        "emitted": segmenter.utterances_emitted,
        "discarded": segmenter.utterances_discarded,
        "effective_max_utterance_ms": segmenter.effective_max_utterance_ms,
        "vad_ms_per_frame": (
            vad_seconds * 1000 / segmenter.frames_pushed if segmenter.frames_pushed else 0.0
        ),
        "wall_clock_s": time.perf_counter() - started,
        "interrupted": interrupted,
        "device_lost": lost,
    }


def render(summary: dict) -> str:
    lines = [
        f"{summary['emitted']} utterances · {summary['discarded']} discarded "
        f"(< min_utterance_ms, a VAD blip not speech — D30)",
        f"  {summary['frames']} frames · {summary['audio_ms'] / 1000:.2f} s audio · "
        f"{summary['speech_ms'] / 1000:.2f} s speech ({summary['speech_ratio']:.0%})",
        f"  VAD cost: {summary['vad_ms_per_frame']:.3f} ms per {cfg.FRAME_MS} ms frame "
        f"(D35 measured 0.156 — this is the number Level 2's acceptance asks for)",
    ]
    if not summary["utterances"]:
        lines.append(
            "  no utterances. Correct for silence or music; if this was speech, "
            "check the peak level and that the right endpoint was captured."
        )
    over = [
        u for u in summary["utterances"]
        if u["duration_ms"] > summary["effective_max_utterance_ms"]
    ]
    if over:
        lines.append(f"  BUG: {len(over)} utterance(s) exceed the max — {over}")
    if summary["interrupted"]:
        lines.append("  stopped by Ctrl-C.")
    if summary["device_lost"]:
        lines.append(f"  DEVICE LOST: {summary['device_lost']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    parser = argparse.ArgumentParser(
        prog="python -m speech_translator.tools.dump_utterances",
        description=(
            "Segment audio into utterances and show where the boundaries landed. "
            "With --write-wav you can listen to each one, which is how "
            "'boundaries land at real pauses' is actually checked."
        ),
    )
    parser.add_argument(
        "--input-wav", type=str, default=None,
        help="read a WAV instead of the device (the only mode that works on Linux)",
    )
    parser.add_argument("-d", "--device", type=int, default=None, help="loopback device index")
    parser.add_argument(
        "--realtime", action="store_true",
        help="with --input-wav, pace at wall-clock speed instead of as fast as possible",
    )
    parser.add_argument(
        "--max-utterance-ms", type=int, default=None,
        help="override the cap at runtime, as the backpressure controller will (D28); "
             "clamped to [max_utterance_floor_ms, max_utterance_ms]",
    )
    parser.add_argument(
        "--write-wav", type=Path, default=None,
        help="write each utterance to its own WAV in this directory",
    )
    parser.add_argument("--json", action="store_true", help="one JSON object per utterance")
    args = parser.parse_args(argv)

    try:
        summary = dump(
            input_wav=args.input_wav,
            device_index=args.device,
            realtime=args.realtime,
            max_utterance_ms=args.max_utterance_ms,
            write_wav=args.write_wav,
            as_json=args.json,
        )
    except AudioSourceError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print(render(summary), file=sys.stderr)
    return 1 if summary["device_lost"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
