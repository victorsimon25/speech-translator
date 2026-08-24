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

#: Generated speech, 13.9 s, 16 kHz mono int16. The only audio in the repo that
#: Silero rates as speech, so it is what makes a *positive* VAD test possible on
#: a machine with no meeting recording (D56). Built by
#: ``speech_translator/tools/make_speech_fixture.py``; provenance and its limits
#: are in ``assets/README.md``.
SPEECH_FIXTURE_PATH = REPO_ROOT / "assets" / "speech_fixture.wav"

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
# Silero's call shape — NOT a change to the frame contract above (D53)
# --------------------------------------------------------------------------

#: Silero v5 wants 64 samples of the *previous* frame prepended to the 512 new
#: ones, so each call is 576 wide. The ONNX input dimension is dynamic, so a
#: 512-wide call runs and returns a plausible-looking number — it is just wrong:
#: real speech reads 0.0005 instead of 1.0. Sources still yield 512-sample
#: frames; the context is carried inside the VAD, alongside the LSTM state.
VAD_CONTEXT_SAMPLES = 64
VAD_INPUT_SAMPLES = VAD_CONTEXT_SAMPLES + FRAME_SAMPLES  # 576
VAD_STATE_SHAPE = (2, 1, 128)

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

    # -- VAD and utterance shaping (D54, D55) ------------------------------
    # TUNABLE, not derived. Unlike the four values above, nothing in D23 or D25
    # fixes these: Silero emits a probability and the design never said where to
    # cut it. They are starting points, chosen with reasons (D54, D55), to be
    # calibrated against real meeting audio at Level 7.
    #: Open an utterance at this probability...
    vad_speech_threshold: float = 0.50
    #: ...and keep it open down to this one. Two thresholds, not one: real
    #: falling edges pass through 0.74 on their way from 1.00 to 0.11, and a
    #: single cut chatters on exactly that frame. Silero's own reference
    #: iterator uses the same 0.15 gap.
    vad_release_threshold: float = 0.35
    #: The `speaking` side channel only — how long speech must be absent before
    #: the UI indicator goes dark. Stops it strobing in inter-word gaps without
    #: making it wait the full `silence_threshold_ms` for the utterance to close.
    #: 160, not 96: measured on the speech fixture, the natural dip around a
    #: plosive runs to three sub-threshold frames, which is exactly where a 96 ms
    #: window expires — the indicator blinked dark for one frame mid-phrase.
    vad_speaking_off_debounce_ms: int = 160
    #: Audio kept from *before* the triggering frame, so Whisper is not handed a
    #: clipped first phoneme.
    utterance_pre_roll_ms: int = 128
    #: Audio kept after the last speech frame. The rest of the closing silence is
    #: dropped — trailing silence is what D30 says Whisper hallucinates onto.
    utterance_tail_pad_ms: int = 192

    # -- ASR ---------------------------------------------------------------
    #: Decided by docs/BENCHMARK.md at Level 3 (D20). Selected: medium at
    #: int8_float16 — RTF 0.488, peak VRAM 1849 MB, no throttling (D60).
    model_size: str | None = "medium"
    compute_type: str | None = "int8_float16"
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

    # -- Language identification (D32, D69) --------------------------------
    #: Speech time, not wall clock — silence must not count toward the window.
    lid_window_speech_ms: int = 10_000
    #: After locking, re-run Whisper without a language pin every N utterances
    #: to check whether the audio language has changed (D69).
    lid_probe_interval: int = 5
    #: How many consecutive probe mismatches (different language with sufficient
    #: confidence) trigger a LID reset so the new language is re-learnt.
    lid_probe_streak: int = 2
    #: Minimum Whisper language probability to count a probe as a mismatch.
    lid_probe_min_confidence: float = 0.7

    # -- Translation (D7, D31, D37) ---------------------------------------
    target_lang: str = "en"
    translate_url: str = "https://api.mymemory.translated.net/get"
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
        vad_speech_threshold=_env_float("VAD_SPEECH_THRESHOLD", 0.50),
        vad_release_threshold=_env_float("VAD_RELEASE_THRESHOLD", 0.35),
        vad_speaking_off_debounce_ms=_env_int("VAD_SPEAKING_OFF_DEBOUNCE_MS", 160),
        utterance_pre_roll_ms=_env_int("UTTERANCE_PRE_ROLL_MS", 128),
        utterance_tail_pad_ms=_env_int("UTTERANCE_TAIL_PAD_MS", 192),
        model_size=_env_opt_str("MODEL_SIZE", "medium"),
        compute_type=_env_opt_str("COMPUTE_TYPE", "int8_float16"),
        device=_env_str("DEVICE", "cuda"),
        model_cache_dir=_env_path("MODEL_CACHE_DIR", REPO_ROOT / "models"),
        asr_queue_maxsize=_env_int("ASR_QUEUE_MAXSIZE", 4),
        no_speech_prob_max=_env_float("NO_SPEECH_PROB_MAX", 0.6),
        avg_logprob_min=_env_float("AVG_LOGPROB_MIN", -1.0),
        lid_window_speech_ms=_env_int("LID_WINDOW_SPEECH_MS", 10_000),
        lid_probe_interval=_env_int("LID_PROBE_INTERVAL", 5),
        lid_probe_streak=_env_int("LID_PROBE_STREAK", 2),
        lid_probe_min_confidence=_env_float("LID_PROBE_MIN_CONFIDENCE", 0.7),
        target_lang=_env_str("TARGET_LANG", "en"),
        mt_char_budget=_env_int("MT_CHAR_BUDGET", 500_000),
        mt_char_hard_stop=_env_int("MT_CHAR_HARD_STOP", 480_000),
        host=_env_str("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        data_dir=_env_path("DATA_DIR", REPO_ROOT / "var"),
        google_translate_api_key=os.environ.get("GOOGLE_TRANSLATE_API_KEY") or None,
    )
