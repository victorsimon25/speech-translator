"""``python -m speech_translator.tools.list_devices``.

Answers "which device will we capture, and in what format" before a recording
is attempted rather than after ten minutes of silence. The rate and channel
count printed here are the ones the source reads at runtime — never hard-coded
(D41) — so if this shows 44 100 Hz, that is what the resampler will be given.
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import config as cfg
from ..audio.source import AudioDeviceUnavailable
from ..audio.wasapi import WasapiLoopbackSource
from ..logging_setup import force_utf8


def collect() -> list[dict]:
    """Loopback devices as plain dicts. Raises on non-Windows (D17, D22)."""
    return [
        {
            "index": int(d["index"]),
            "name": str(d["name"]),
            "rate": int(d["defaultSampleRate"]),
            "channels": int(d["maxInputChannels"]),
            "is_default_output": bool(d.get("is_default_output")),
        }
        for d in WasapiLoopbackSource.loopback_devices()
    ]


def render(devices: list[dict]) -> str:
    if not devices:
        return "No WASAPI loopback devices. Is any audio endpoint active?"
    width = max(len(d["name"]) for d in devices)
    lines = [
        f"WASAPI loopback devices · capture target is {cfg.SAMPLE_RATE} Hz mono "
        f"int16, {cfg.FRAME_SAMPLES}-sample frames",
        "",
    ]
    for d in devices:
        mark = "->" if d["is_default_output"] else "  "
        lines.append(
            f"{mark} [{d['index']:>2}] {d['name'].ljust(width)}  "
            f"{d['rate']} Hz · {d['channels']} ch"
        )
    lines.append("")
    lines.append(
        "'->' is the default output device — what you actually hear, and what "
        "record_loopback picks with no --device."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    parser = argparse.ArgumentParser(
        prog="python -m speech_translator.tools.list_devices",
        description="List WASAPI loopback capture devices (Windows only).",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        devices = collect()
    except AudioDeviceUnavailable as exc:
        # Honest rather than pretending: this command needs the demo machine.
        print(f"{exc}", file=sys.stderr)
        return 2

    print(json.dumps(devices, ensure_ascii=False, indent=2) if args.json else render(devices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
