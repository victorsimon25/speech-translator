# Vendored assets

## `silero_vad.onnx`

Silero VAD, ONNX build, **vendored deliberately** — the `silero-vad` PyPI
package imports `torch` unconditionally at module scope, including on its
`onnx=True` path, and a second CUDA runtime alongside CTranslate2 on a 4 GB card
is a real failure rather than a theoretical one. See **D35**. Never add the
package as a dependency; load this file with `onnxruntime` instead.

| | |
|---|---|
| Source | `silero_vad` 6.2.1, `silero_vad/data/silero_vad.onnx` |
| License | MIT (`silero_vad.LICENSE`, Silero Team) |
| Size | 2 327 524 bytes |
| SHA-256 | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |

Model interface, confirmed against this file:

```
inputs : input [N, 512] float32 · state [2, N, 128] float32 · sr int64
outputs: output [N, 1] float32  · stateN [2, N, 128] float32
```

State is carried between frames — it is not stateless per frame. Measured cost
**0.156 ms per 32 ms frame** (D35); `doctor` re-measures it on whichever machine
it runs on.
