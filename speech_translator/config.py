"""Configuration.

Two rules about this file:

1. **The segmentation values are derived, not defaults.** ``silence_threshold_ms``
   and ``max_utterance_ms`` come out of the latency budget in D25 —
   ``word_age = silence + MT + D x (1 + RTF)`` against a p95 target of 7 s.
   ``max_utterance_ms`` is the dominant latency knob, not RTF. Read D25 before
   changing either.
2. **``model_size`` and ``compute_type`` are None on purpose.** They are decided
   by the measurement in ``docs/BENCHMARK.md``, run on the Windows machine at
   Level 3, and written in here afterwards. Do not hard-code a guess (D20).

Every value can be overridden from the environment with an ``ST_`` prefix, so
the demo machine can be tuned without editing tracked files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

#: Vendored Silero weights (2.3 MB, MIT). The ``silero-vad`` package is never
#: imported — it pulls torch unconditionally (D35).
SILERO_ONNX_PATH = REPO_ROOT / "assets" / "silero_vad.onnx"

# --------------------------------------------------------------------------
# Audio contract — fixed everywhere, never varied (D23, D24)
# --------------------------------------------------------------------------

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # signed 16-bit little-endian
FRAME_SAMPLES = 512  # Silero's required size at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * CHANNELS * SAMPLE_WIDTH_BYTES  # 1024
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32

# --------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------

ENV_PREFIX = "ST_"


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value if value not in (None, "") else None


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return default if raw is None else float(raw)


def _env_str(name: str, default: str) -> str:
    raw = _env(name)
    return default if raw is None else raw


def _env_opt_str(name: str, default: str | None) -> str | None:
    raw = _env(name)
    return default if raw is None else raw


def _env_path(name: str, default: Path) -> Path:
    raw = _env(name)
    return default if raw is None else Path(raw).expanduser()


@dataclass(frozen=True)
class Config:
    # -- Segmentation: derived from the D25 latency budget, not guessed ----
    silence_threshold_ms: int = 600
    max_utterance_ms: int = 4_000
    min_utterance_ms: int = 300
    max_utterance_floor_ms: int = 2_000

    # -- ASR ---------------------------------------------------------------
    #: Decided by docs/BENCHMARK.md at Level 3 (D20). None until then, and the
    #: transcriber must refuse to start rather than substitute a default.
    model_size: str | None = None
    compute_type: str | None = None
    device: str = "cuda"  # no CPU fallback is built (D36)
    beam_size: int = 1
    condition_on_previous_text: bool = False
    vad_filter: bool = False  # segmentation is ours; Whisper's VAD is off
    #: Pinned so first-run weight download lands somewhere known (D41).
    model_cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "models")

    #: Bounded Segmenter -> Transcriber queue, ~16 s of backlog (D28).
    asr_queue_maxsize: int = 4

    # -- Hallucination guard (D30) ----------------------------------------
    # TUNABLE, not derived: unlike the D25 values above, these two are starting
    # points to be calibrated against real transcripts at Level 4/7.
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.0

    # -- Backpressure (D28) ------------------------------------------------
    shrink_trigger_depth: int = 2
    shrink_after_utterances: int = 3
    shrink_factor: float = 0.75
    recover_rtf_ema: float = 0.35
    recover_after_s: int = 30

    # -- Language identification (D32) ------------------------------------
    #: Speech time, not wall clock — silence must not count toward the window.
    lid_window_speech_ms: int = 10_000

    # -- Translation (D7, D31, D37) ---------------------------------------
    target_lang: str = "en"
    translate_url: str = "https://translation.googleapis.com/language/translate/v2"
    translate_timeout_s: float = 5.0
    translate_retries: int = 2
    translate_cache_size: int = 512
    #: 500 000 chars/month free, no roll-over. Warn at 80%; hard-stop below the
    #: free ceiling so an overrun cannot become a paid call.
    mt_char_budget: int = 500_000
    mt_char_warn_ratio: float = 0.8
    mt_char_hard_stop: int = 480_000

    # -- Server (D38) ------------------------------------------------------
    host: str = "127.0.0.1"
    #: Config, not a literal — "port already in use" is on the D41 register.
    port: int = 8000

    # -- State and evidence ------------------------------------------------
    #: Persisted month-keyed character counter (D31) and the per-session JSONL
    #: timing log that is the evidence base for the DoD (D34).
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "var")

    # -- Secrets (never written to a tracked file) ------------------------
    google_translate_api_key: str | None = None

    # -- Derived -----------------------------------------------------------
    @property
    def flush_silence_ms(self) -> int:
        """A held sentence fragment is flushed after this much silence (D29)."""
        return 2 * self.silence_threshold_ms

    @property
    def mt_budget_file(self) -> Path:
        return self.data_dir / "mt_chars.json"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def predicted_word_age_ms(self, rtf: float, mt_ms: int = 300) -> float:
        """The D25 budget, evaluated. Prediction only — Level 7 measures it."""
        return (
            self.silence_threshold_ms
            + mt_ms
            + self.max_utterance_ms * (1.0 + rtf)
        )


def load_config() -> Config:
    """Build a Config from defaults overlaid with .env and the environment."""
    load_dotenv(REPO_ROOT / ".env", override=False)

    return Config(
        silence_threshold_ms=_env_int("SILENCE_THRESHOLD_MS", 600),
        max_utterance_ms=_env_int("MAX_UTTERANCE_MS", 4_000),
        min_utterance_ms=_env_int("MIN_UTTERANCE_MS", 300),
        max_utterance_floor_ms=_env_int("MAX_UTTERANCE_FLOOR_MS", 2_000),
        model_size=_env_opt_str("MODEL_SIZE", None),
        compute_type=_env_opt_str("COMPUTE_TYPE", None),
        device=_env_str("DEVICE", "cuda"),
        model_cache_dir=_env_path("MODEL_CACHE_DIR", REPO_ROOT / "models"),
        asr_queue_maxsize=_env_int("ASR_QUEUE_MAXSIZE", 4),
        no_speech_prob_max=_env_float("NO_SPEECH_PROB_MAX", 0.6),
        avg_logprob_min=_env_float("AVG_LOGPROB_MIN", -1.0),
        lid_window_speech_ms=_env_int("LID_WINDOW_SPEECH_MS", 10_000),
        target_lang=_env_str("TARGET_LANG", "en"),
        mt_char_budget=_env_int("MT_CHAR_BUDGET", 500_000),
        mt_char_hard_stop=_env_int("MT_CHAR_HARD_STOP", 480_000),
        host=_env_str("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        data_dir=_env_path("DATA_DIR", REPO_ROOT / "var"),
        google_translate_api_key=os.environ.get("GOOGLE_TRANSLATE_API_KEY") or None,
    )
