"""Stage 2 of the pipeline: frames in, utterances out.

`INTERFACES.md` §2. Silero over the vendored ONNX (never the `silero-vad`
package, never torch — D35), and a state machine that closes an utterance on
silence or on length, whichever comes first (D12) at the D25 values.

Importing this package is cheap: `onnxruntime` is imported inside `SileroVad`,
so nothing is loaded until a VAD is actually constructed.
"""

from __future__ import annotations

from .segmenter import BYTES_PER_MS, Segmenter, Utterance
from .vad import SileroVad, SpeechDetector

__all__ = [
    "BYTES_PER_MS",
    "Segmenter",
    "SileroVad",
    "SpeechDetector",
    "Utterance",
]
