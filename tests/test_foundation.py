"""Level 0 acceptance, as tests.

These run on Linux with no audio hardware, no GPU and no network — that is the
point: the dev box must be able to prove the foundation before anything is
pushed to the Windows machine (D22).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

import speech_translator
from speech_translator import config as cfg


def test_package_imports_on_this_platform():
    """D22: no eager Windows imports. Importing must work on Linux."""
    assert speech_translator.__version__


def test_no_torch_anywhere():
    """D35: torch would put a second CUDA runtime on a 4 GB card."""
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("torchaudio") is None


def test_silero_vad_package_is_not_a_dependency():
    """D35: the weights are vendored; the package is never imported."""
    assert importlib.util.find_spec("silero_vad") is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("silence_threshold_ms", 600),
        ("max_utterance_ms", 4_000),
        ("min_utterance_ms", 300),
        ("max_utterance_floor_ms", 2_000),
        ("asr_queue_maxsize", 4),
    ],
)
def test_d25_values_are_the_derived_ones(key, value):
    """These are derived from the latency budget, not defaults (D25)."""
    assert getattr(cfg.Config(), key) == value


def test_model_choice_is_deferred_to_the_benchmark():
    """D20: Level 3 decides these. A default here would be a guess."""
    c = cfg.Config()
    assert c.model_size is None
    assert c.compute_type is None


def test_config_satisfies_the_word_age_target():
    """D25/D26: the config must predict p95 word age < 7 s at the RTF gate."""
    c = cfg.Config()
    assert c.predicted_word_age_ms(rtf=0.5) / 1000 < 7.0


def test_flush_rule_threshold():
    """D29: a held fragment flushes after 2x the silence threshold."""
    c = cfg.Config()
    assert c.flush_silence_ms == 2 * c.silence_threshold_ms


def test_audio_contract_constants():
    """D23/D24: 16 kHz mono int16, 512-sample frames, never varied."""
    assert (cfg.SAMPLE_RATE, cfg.CHANNELS, cfg.SAMPLE_WIDTH_BYTES) == (16_000, 1, 2)
    assert cfg.FRAME_SAMPLES == 512
    assert cfg.FRAME_BYTES == 1024
    assert cfg.FRAME_MS == 32


def test_vendored_onnx_present_and_runs_one_frame():
    """D35/D53: the file is in the repo and matches the *corrected* signature.

    This test used to pass `FRAME_SAMPLES` (512) and assert only the output
    shapes. It passed, and the VAD it was checking could not detect speech: the
    input dimension is dynamic, so the wrong width runs and returns 0.0005 on
    speech. Shapes are not an answer. `tests/test_segment.py` asserts the answer;
    this one now at least calls the model the way the model is documented.
    """
    import numpy as np
    import onnxruntime as ort

    assert cfg.SILERO_ONNX_PATH.exists()

    sess = ort.InferenceSession(str(cfg.SILERO_ONNX_PATH), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    assert names == {"input", "state", "sr"}

    assert cfg.VAD_INPUT_SAMPLES == cfg.VAD_CONTEXT_SAMPLES + cfg.FRAME_SAMPLES == 576
    out, state = sess.run(
        None,
        {
            "input": np.zeros((1, cfg.VAD_INPUT_SAMPLES), dtype=np.float32),
            "state": np.zeros(cfg.VAD_STATE_SHAPE, dtype=np.float32),
            "sr": np.array(cfg.SAMPLE_RATE, dtype=np.int64),
        },
    )
    assert out.shape == (1, 1)
    assert state.shape == cfg.VAD_STATE_SHAPE


def test_utf8_is_forced_and_non_ascii_survives_a_subprocess():
    """D41: the Windows console is cp1252; a caption is non-ASCII.

    Forcing a cp1252 stdout in the child reproduces the Windows console on
    Linux — without force_utf8() this raises UnicodeEncodeError in the child.
    """
    code = (
        "import sys;"
        "sys.stdout.reconfigure(encoding='cp1252');"
        "import speech_translator;"
        "print('¿Qué? こんにちは')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=cfg.REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "¿Qué?" in proc.stdout.decode("utf-8")


def test_doctor_runs_and_reports_without_raising():
    """D40: doctor must report, never blow up — including with no GPU."""
    from speech_translator import doctor

    results = doctor.run_checks(cfg.load_config(), allow_network=False)
    names = {r.name for r in results}
    assert {"CUDA visible to CTranslate2", "WASAPI loopback device"} <= names
    assert all(r.status in {"PASS", "FAIL", "WARN", "SKIP"} for r in results)
    assert doctor.render(results)


@pytest.mark.skipif(sys.platform == "win32", reason="Linux dev-box expectation")
def test_cuda_and_loopback_fail_but_are_expected_on_linux():
    """The documented Linux behaviour: FAIL, but not a failing exit code."""
    from speech_translator import doctor

    results = doctor.run_checks(cfg.load_config(), allow_network=False)
    by_name = {r.name: r for r in results}
    for name in ("CUDA visible to CTranslate2", "WASAPI loopback device"):
        assert by_name[name].status == "FAIL"
        assert by_name[name].expected_failure
