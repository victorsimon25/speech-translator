"""Silero VAD over the vendored ONNX — one probability per 32 ms frame.

`INTERFACES.md` §2. Never import the `silero-vad` package and never import
torch: the package pulls torch unconditionally, and a second CUDA runtime beside
CTranslate2 on a 4 GB card is a real failure, not a theoretical one (D35). This
module talks to `onnxruntime` directly and loads `assets/silero_vad.onnx`.

**Two things carry across the frame boundary, not one** (D53):

1. the LSTM `state`, shape `[2, 1, 128]`, threaded call to call; and
2. **64 samples of the previous frame's audio**, prepended to the 512 new ones,
   so every call is 576 wide.

The second one is the trap. The ONNX input dimension is dynamic, so passing 512
samples runs cleanly and returns a plausible-looking float — it just returns
**0.0005 on real recorded speech** where the correct call returns **1.0**. A VAD
that answers "no speech, ever" produces no utterances, no transcripts and no
captions, and nothing in the stack raises. Both pieces of state have a guard in
`tests/test_segment.py`, each with a control proving the guard can see the bug it
exists for — the same discipline D46 applied to the resampler's delay line, which
is the same class of failure one layer down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .. import config as cfg

#: Scale int16 to the float range the model was trained on. 32768, not 32767:
#: it is the magnitude of the most negative sample, so nothing can exceed 1.0.
_INT16_FULL_SCALE = 32768.0


@runtime_checkable
class SpeechDetector(Protocol):
    """What `Segmenter` needs from a VAD, and nothing more.

    Deliberately narrow, because the segmenter's state machine is arithmetic and
    should be tested with scripted probabilities rather than by hoping a neural
    network fires on a synthetic tone. `tests/test_segment.py` implements this
    with a list of numbers; production implements it with Silero.
    """

    def probability(self, frame: bytes) -> float:
        """Speech probability in [0, 1] for one 512-sample frame."""
        ...

    def reset(self) -> None:
        """Forget everything carried between frames. New stream, clean state."""
        ...


class SileroVad:
    """The vendored Silero v5 ONNX, one frame at a time, state carried."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        sample_rate: int = cfg.SAMPLE_RATE,
    ) -> None:
        import onnxruntime as ort  # local: keeps the module cheap to import

        self.model_path = Path(model_path or cfg.SILERO_ONNX_PATH)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"vendored Silero model missing: {self.model_path}. "
                f"It is committed to the repo (see assets/README.md); do not "
                f"pip install silero-vad as a substitute (D35)."
            )
        self.sample_rate = int(sample_rate)

        # One thread on purpose. The model costs ~0.12 ms per 32 ms frame, so a
        # thread pool cannot pay for its own synchronisation, and on the demo
        # machine these cores are shared with capture, the server, the browser
        # and CTranslate2's host side. CPU provider only — the GPU has 4 GB and
        # a transcriber to run (D18).
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )

        self._sr = np.array(self.sample_rate, dtype=np.int64)
        self.reset()

    # -- SpeechDetector ----------------------------------------------------

    def reset(self) -> None:
        self._state = np.zeros(cfg.VAD_STATE_SHAPE, dtype=np.float32)
        # Zeros for the first frame: at t=0 there is no previous audio, which is
        # what silence looks like anyway.
        self._context = np.zeros(cfg.VAD_CONTEXT_SAMPLES, dtype=np.float32)
        self.frames_seen = 0

    def probability(self, frame: bytes) -> float:
        if len(frame) != cfg.FRAME_BYTES:
            raise ValueError(
                f"expected a {cfg.FRAME_BYTES}-byte frame "
                f"({cfg.FRAME_SAMPLES} samples), got {len(frame)} bytes. "
                f"Frame size is fixed by the VAD and never varied (D23)."
            )

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        samples /= _INT16_FULL_SCALE

        # The 576-wide call: 64 samples of history, then the 512 new ones (D53).
        model_input = np.concatenate((self._context, samples))[np.newaxis, :]
        output, self._state = self._session.run(
            None, {"input": model_input, "state": self._state, "sr": self._sr}
        )
        # Tail of *this* frame becomes the next call's context. Taken from the
        # new samples, not from `model_input`, so it cannot silently degenerate
        # into re-feeding old context forever.
        self._context = samples[-cfg.VAD_CONTEXT_SAMPLES :]
        self.frames_seen += 1
        return float(output[0, 0])

    def describe(self) -> str:
        return (
            f"silero v5 · {self.model_path.name} · "
            f"{cfg.VAD_INPUT_SAMPLES}-sample calls "
            f"({cfg.VAD_CONTEXT_SAMPLES} context + {cfg.FRAME_SAMPLES} new) · "
            f"{self.sample_rate} Hz · CPU, 1 thread"
        )
