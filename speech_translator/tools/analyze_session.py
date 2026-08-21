"""``python -m speech_translator.tools.analyze_session``.

Reads one session's JSONL logs and prints a latency/quality summary:

  - p50 / p95 / max word_age_ms  (gate 3: < 7000 ms)
  - RTF curve: first minute vs last minute vs full session
  - Queue-depth histogram
  - Utterance accept/reject breakdown
  - Characters translated and monthly budget usage

Two log files per session are expected (both written by the app):

  var/logs/asr_<timestamp>.jsonl      — one record per utterance (worker)
  var/logs/session_<timestamp>.jsonl  — caption + health records (publisher)

If only the ASR log is available (e.g. no API key, no translation), word_age is
computed from the formula using --mt-ms (default 300 ms) for the MT term.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Percentile helper (no numpy)
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _ts_to_epoch(ts: str) -> float:
    """Parse an ISO timestamp string to a Unix float."""
    try:
        # Python 3.11+ fromisoformat handles 'Z'; earlier needs a workaround.
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Treat naive timestamps as local time — worker uses datetime.now()
            return dt.timestamp()
        return dt.timestamp()
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _discover_pair(log_dir: Path) -> tuple[Path | None, Path | None]:
    """Find the most recent asr_*.jsonl and its matching session_*.jsonl."""
    asr_files = sorted(log_dir.glob("asr_*.jsonl"), key=lambda p: p.name, reverse=True)
    if not asr_files:
        return None, None
    asr = asr_files[0]
    # Extract timestamp suffix (e.g. "20260821-143022") from "asr_20260821-143022.jsonl"
    stem = asr.stem  # "asr_YYYYMMDD-HHMMSS"
    suffix = stem[len("asr_"):]
    session_candidate = log_dir / f"session_{suffix}.jsonl"
    session = session_candidate if session_candidate.exists() else None
    return asr, session


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _gate(value: float, threshold: float) -> str:
    return "PASS ✓" if value < threshold else "FAIL ✗"


def analyze(
    asr_path: Path,
    session_path: Path | None,
    mt_ms_default: int,
    config_silence_ms: int = 600,
) -> str:
    asr_records = _load_jsonl(asr_path)
    session_records = _load_jsonl(session_path) if session_path else []

    # --- MT latency per utterance from session log ---------------------------
    # Session log has caption records; take the first sentence per utterance.
    mt_per_utterance: dict[str, int] = {}
    for r in session_records:
        if r.get("type") == "caption":
            uid = r.get("utterance_id", "")
            if uid and uid not in mt_per_utterance:
                mt_per_utterance[uid] = int(r.get("mt_ms", mt_ms_default))

    mt_source = (
        f"measured, n={len(mt_per_utterance)}"
        if mt_per_utterance
        else f"assumed (--mt-ms {mt_ms_default})"
    )

    # --- Parse ASR records ---------------------------------------------------
    all_records = [r for r in asr_records if isinstance(r, dict) and "audio_ms" in r]
    if not all_records:
        return f"No utterance records found in {asr_path}"

    # Timestamps for minute-splitting
    ts_values = [_ts_to_epoch(r["ts"]) for r in all_records if "ts" in r]
    has_ts = len(ts_values) == len(all_records)

    if has_ts:
        t_start = ts_values[0]
        t_end = ts_values[-1]
        t_duration_s = t_end - t_start
    else:
        t_start = 0.0
        t_end = 0.0
        t_duration_s = 0.0

    total = len(all_records)
    accepted_recs = [r for r in all_records if r.get("accepted", False)]
    rejected_count = total - len(accepted_recs)

    # --- Word age (accepted utterances only) ---------------------------------
    word_ages: list[float] = []
    for r in accepted_recs:
        audio_ms = r.get("audio_ms", 0)
        asr_ms = r.get("asr_ms", 0)
        uid = r.get("utterance_id", "")
        mt_ms = mt_per_utterance.get(uid, mt_ms_default)
        wa = config_silence_ms + mt_ms + audio_ms + asr_ms
        word_ages.append(float(wa))

    wa_p50 = _percentile(word_ages, 50)
    wa_p95 = _percentile(word_ages, 95)
    wa_max = _percentile(word_ages, 100)

    # --- RTF by minute -------------------------------------------------------
    def rtf_stats(recs: list[dict]) -> str:
        rtfs = [float(r["rtf"]) for r in recs if "rtf" in r]
        if not rtfs:
            return "n/a"
        return (
            f"p50 {_percentile(rtfs, 50):.3f}  "
            f"p95 {_percentile(rtfs, 95):.3f}  "
            f"mean {_mean(rtfs):.3f}  n={len(rtfs)}"
        )

    if has_ts and t_duration_s > 0:
        first_min = [r for r, t in zip(all_records, ts_values) if t - t_start < 60.0]
        last_min = [r for r, t in zip(all_records, ts_values) if t_end - t < 60.0]
    else:
        first_min = all_records[:max(1, total // 10)]
        last_min = all_records[-max(1, total // 10):]

    # --- Queue depth histogram -----------------------------------------------
    depth_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 3 = "≥3"
    depth_from_asr = [r for r in all_records if "queue_depth" in r]
    for r in depth_from_asr:
        d = int(r["queue_depth"])
        key = min(d, 3)
        depth_counts[key] = depth_counts.get(key, 0) + 1

    # Queue depth from health records in session log
    health_depths: list[int] = [
        int(r["queue_depth"])
        for r in session_records
        if r.get("type") == "health" and "queue_depth" in r
    ]
    health_depth_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for d in health_depths:
        key = min(d, 3)
        health_depth_counts[key] = health_depth_counts.get(key, 0) + 1

    depth_source_count = len(depth_from_asr) or len(health_depths)
    depth_source = "ASR log" if depth_from_asr else "health log" if health_depths else "unavailable"
    if depth_from_asr:
        depth_data = depth_counts
        depth_total = len(depth_from_asr)
    elif health_depths:
        depth_data = health_depth_counts
        depth_total = len(health_depths)
    else:
        depth_data = {0: 0, 1: 0, 2: 0, 3: 0}
        depth_total = 0

    queue_ok = (depth_data.get(2, 0) + depth_data.get(3, 0)) == 0

    # --- Characters ----------------------------------------------------------
    total_chars_session = sum(
        int(r.get("chars", 0))
        for r in session_records
        if r.get("type") == "caption"
    )

    # Monthly budget file
    budget_str = "n/a"
    try:
        from .. import config as cfg
        c = cfg.load_config()
        month_key = datetime.now().strftime("%Y_%m")
        budget_file = c.data_dir / "usage" / f"chars_{month_key}.json"
        if budget_file.exists():
            with open(budget_file, encoding="utf-8") as bf:
                budget_data = json.load(bf)
            chars_used = budget_data.get("chars_used", 0)
            chars_budget = c.mt_char_budget
            budget_str = f"{chars_used:,} / {chars_budget:,}  ({100 * chars_used / chars_budget:.1f}%)"
    except Exception:
        pass

    # --- Build output --------------------------------------------------------
    sep = "=" * 60
    lines = [
        sep,
        "  Session Analysis",
        sep,
        f"  ASR log:      {asr_path.name}",
    ]
    if session_path:
        lines.append(f"  Session log:  {session_path.name}")
    if has_ts and t_duration_s > 0:
        m, s = divmod(int(t_duration_s), 60)
        lines.append(f"  Duration:     {m}:{s:02d} min")
    lines += [
        f"  Utterances:   {total} total  |  {len(accepted_recs)} accepted  |  {rejected_count} rejected",
        "",
        "--- Word Age  (formula: silence + MT + audio_ms + asr_ms) ---",
        f"  p50:     {wa_p50:,.0f} ms",
        f"  p95:     {wa_p95:,.0f} ms   {_gate(wa_p95, 7000)}  (gate 3: < 7000 ms)",
        f"  max:     {wa_max:,.0f} ms",
        f"  MT term: {mt_source}",
        "",
        "--- RTF Distribution ---",
        f"  First minute ({len(first_min)} utterances):  {rtf_stats(first_min)}",
        f"  Last  minute ({len(last_min)} utterances):  {rtf_stats(last_min)}",
        f"  Full session ({len(all_records)} utterances):  {rtf_stats(all_records)}",
        "",
        f"--- Queue Depth  (source: {depth_source},  n={depth_total}) ---",
    ]
    if depth_total > 0:
        for key, label in [(0, "0 (empty)"), (1, "1"), (2, "2"), (3, "≥3")]:
            count = depth_data.get(key, 0)
            pct = 100.0 * count / depth_total
            lines.append(f"  {label:<10}  {count:5d}  ({pct:5.1f}%)")
        lines.append(
            f"  Gate (0-1):  {'PASS ✓' if queue_ok else 'FAIL ✗'}"
            f"  ({'no depth ≥2' if queue_ok else str(depth_data.get(2,0)+depth_data.get(3,0))+' events ≥2'})"
        )
    else:
        lines.append("  (no queue_depth data — upgrade the server to add it)")
    lines += [
        "",
        "--- Hallucination Guard ---",
        f"  Accepted:  {len(accepted_recs):5d}  ({100 * len(accepted_recs) / max(total, 1):.1f}%)",
        f"  Rejected:  {rejected_count:5d}  ({100 * rejected_count / max(total, 1):.1f}%)",
        "",
        "--- Characters ---",
    ]
    if total_chars_session:
        lines.append(f"  This session:  {total_chars_session:,} chars")
    lines.append(f"  Monthly total: {budget_str}")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m speech_translator.tools.analyze_session",
        description="Analyse a session's JSONL logs and print a latency summary.",
    )
    parser.add_argument(
        "--asr", type=Path, default=None,
        help="path to asr_<timestamp>.jsonl (explicit; overrides --dir)",
    )
    parser.add_argument(
        "--session", type=Path, default=None,
        help="path to session_<timestamp>.jsonl (explicit; overrides --dir)",
    )
    parser.add_argument(
        "--dir", type=Path, default=None,
        help="directory to auto-discover the latest asr + session log pair",
    )
    parser.add_argument(
        "--mt-ms", type=int, default=300,
        help="assumed MT round-trip in ms when no session log is available (default: 300)",
    )
    args = parser.parse_args(argv)

    asr_path = args.asr
    session_path = args.session

    if asr_path is None:
        log_dir = args.dir
        if log_dir is None:
            # Fall back to project default.
            try:
                from .. import config as cfg
                log_dir = cfg.load_config().logs_dir
            except Exception:
                log_dir = Path("var") / "logs"
        asr_path, session_path_auto = _discover_pair(log_dir)
        if session_path is None:
            session_path = session_path_auto

    if asr_path is None or not asr_path.exists():
        print(
            f"No ASR log found. Run a session first, then pass --asr or --dir.",
            file=sys.stderr,
        )
        return 2

    print(analyze(asr_path, session_path, mt_ms_default=args.mt_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
