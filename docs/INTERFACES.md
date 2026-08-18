# Interfaces

The contract between stages. **A session working on one stage should be able to
read only this file plus that stage's own source.** Keep it that way — if you
find yourself reading another stage's implementation to understand its output,
this file is wrong and should be fixed.

## Pipeline

```
AudioSource → Segmenter → Transcriber → SentenceSplitter → Translator → Publisher → UI
 (WASAPI)      (Silero VAD  (Whisper,      (carry-over       (Google      (WebSocket)  (browser)
               + max-len)    local CUDA)    fragment)         Cloud)
                           └── worker process ──┘
```

Every arrow is a queue. The `Segmenter → Transcriber` queue crosses a **process**
boundary (D15); the Transcriber's return path crosses back. Everything except the
Transcriber lives in the main asyncio process — including translation, which is
network-bound and would otherwise idle the GPU (D27). The rest may be threads or
coroutines.

**The `Segmenter → Transcriber` queue is bounded** (`asr_queue_maxsize = 4`) with
an adaptive-shrink-then-drop-oldest policy. See D28; the depth is published in
the `health` message.

---

## 1. AudioSource

Yields raw PCM frames. Platform-specific; everything downstream is not.

```python
class AudioSource(Protocol):
    def frames(self) -> Iterator[bytes]: ...
    def close(self) -> None: ...
```

**Frame format — fixed across all implementations:**
- 16 000 Hz, mono, signed 16-bit little-endian PCM
- **512-sample frames (32 ms).** Settled: the VAD is Silero, which wants 512
  samples at 16 kHz (D23). Do not vary it.
- Normalisation is the **source's** job (D24). WASAPI loopback delivers the
  device's native format — usually 48 kHz stereo — so downmix and resample
  happen inside the source. No downstream stage contains a sample-rate branch.

**Implementations:**

| Class | Platform | Backing |
|---|---|---|
| `PipeWireMonitorSource` | Linux | **Not implemented (D17).** Path verified available in D2; descoped when Windows became the target |
| `WasapiLoopbackSource` | Windows | `PyAudioWPatch` WASAPI loopback (D3). **The live source.** Import is lazy so the package still imports on Linux (D22) |
| `WavFileSource` | any | Reads a WAV at wall-clock speed, or as fast as possible with `realtime=False`. **Use this for tests** and as the Linux smoke path (D22) |
| `MicrophoneSource` | future | Only if two-way is required (D4) |

Source selection happens once at startup and is the *only* place platform
detection is allowed to appear — `speech_translator.audio.open_source()` is that
place.

**Where normalisation lives.** Both implementations own a `FrameFormatter`
(`speech_translator/audio/format.py`), which is the decode → downmix → streaming
resample → framing chain and nothing else. It is internal to this stage: no
downstream module imports it. Two properties worth knowing from outside:

- **One `soxr.ResampleStream` per source, never rebuilt** — the filter state has
  to carry across chunks or every chunk boundary becomes a click (D46).
- **16 kHz mono int16 in is bit-exact out** — no resampler is constructed at
  all, so replaying a `record_loopback` capture is byte-identical to the live
  capture rather than resampled twice.

The final frame of a stream is zero-padded to 512 samples rather than dropped,
so total sample count matches source duration to within one frame.

---

## 2. Segmenter

Consumes frames, applies voice-activity detection, emits complete utterances.

```python
@dataclass
class Utterance:
    id: str
    pcm: bytes                                  # 16 kHz mono int16
    start_ms: int                               # since session start
    end_ms: int
    closed_by: Literal["silence", "max_length"]
    preceding_silence_ms: int                   # gap before this utterance
```

**Closing rule (D12):** close when silence exceeds `silence_threshold_ms`, **or**
when the buffer reaches `max_utterance_ms` — whichever comes first.

**Tunables — derived, not guessed (D25).** These come out of the latency budget:

| Key | Value | Why |
|---|---|---|
| `silence_threshold_ms` | 600 | latency budget |
| `max_utterance_ms` | 4000 | largest `D` meeting a 7 s word-age target at the RTF gate |
| `min_utterance_ms` | 300 | below this, discard — VAD blip, not speech (D30) |
| `max_utterance_floor_ms` | 2000 | floor for adaptive shrink under load (D28) |

**Tunables — TUNABLE, not derived (D54, D55).** Nothing in D23 or D25 fixes
these; Silero emits a probability and the design never said where to cut it.
They are starting points with reasons, to be calibrated at Level 7:

| Key | Value | Why |
|---|---|---|
| `vad_speech_threshold` | 0.50 | opens an utterance |
| `vad_release_threshold` | 0.35 | keeps it open — hysteresis, so a mid-word dip does not chatter |
| `vad_speaking_off_debounce_ms` | 160 | the `speaking` indicator only |
| `utterance_pre_roll_ms` | 128 | audio kept from *before* the trigger, so the first phoneme is not clipped |
| `utterance_tail_pad_ms` | 192 | audio kept after the last speech frame; the rest of the closing silence is dropped |

**Consequences of the padding, which callers can rely on:**

- `len(pcm) == (end_ms - start_ms) * 32` for every utterance.
- `preceding_silence_ms == start_ms - previous_emitted.end_ms`, always — the gap
  accumulates across a discarded blip, so it stays reconstructible from the
  emitted fields alone.
- The `min_utterance_ms` discard measures **speech**, between the first and last
  speech frames, not `end_ms - start_ms`. The padding is 320 ms on its own, which
  is more than `min_utterance_ms`; testing the padded duration would let every
  blip through. So every utterance published here contains at least
  `min_utterance_ms` of speech — strictly stronger than the Transcriber's own
  check in §3.
- A `max_length` close **reopens immediately and contiguously**: the next
  utterance begins on the very next frame, with `preceding_silence_ms = 0` and no
  pre-roll. §4's fragment carry-over depends on that audio being unbroken.

`max_utterance_ms` is the **dominant latency knob** — see D25 before changing it.
The Segmenter must accept a *runtime* override of the effective value, because
the backpressure controller shrinks it under load (D28).

`preceding_silence_ms` is carried so the silence-visualisation bonus is cheap if
that turns out to be what the brief means. It costs nothing to populate.

The Segmenter also emits a boolean `speaking` signal for the live UI indicator
(D11). That is a side channel, not part of `Utterance`: `Segmenter.speaking` for
polling, and an `on_speaking(bool)` callback that fires **on transitions only** —
31 messages a second down a WebSocket is a flood, not a side channel. It follows
the debounced VAD state, not the utterance, so the dot goes dark ~160 ms after
speech stops rather than waiting the 600 ms the utterance needs.

**VAD runtime.** Silero from a **vendored `silero_vad.onnx`** (2.3 MB, in the
repo) via `onnxruntime`. The `silero-vad` PyPI package is *not* a dependency —
it imports torch unconditionally, which would put a second CUDA runtime on a
4 GB card (D35).

```
inputs : input [N, 576] float32 · state [2, N, 128] float32 · sr int64
outputs: output [N, 1] float32  · stateN [2, N, 128] float32
```

**576, not 512 — corrected in D53.** This is Silero v5: each call takes the 512
new samples with **64 samples of the previous frame prepended**. So *two* things
carry across the frame boundary, not one — the LSTM `state`, and a 64-sample
audio context. The ONNX input dimension is dynamic, so a 512-wide call runs
cleanly and returns a plausible float; it returns **0.0005 on real speech** where
the correct call returns **1.0**, and nothing anywhere raises. This document said
512 until D53 measured it, and so did `doctor`, which passed the dead VAD.

Frames on the wire are still 512 samples (D23) — the context lives inside the
VAD, not in the `AudioSource` contract. Measured cost at the correct width:
**0.116–0.141 ms per 32 ms frame**, inside D35's 0.156 budget.

---

## 3. Transcriber

```python
@dataclass
class Transcript:
    utterance_id: str
    text: str                  # punctuated — Whisper emits punctuation natively
    language: str              # ISO 639-1, from Whisper's own LID
    language_confidence: float
    asr_ms: int                # wall-clock; used to compute RTF at runtime
    no_speech_prob: float      # hallucination guard (D30)
    avg_logprob: float         # hallucination guard (D30)
```

- Model size, `device` and `compute_type` are all decided by `BENCHMARK.md` and
  read from config. Do not hard-code them before that runs (D20).
- `language` is populated from Whisper's built-in language identification — no
  separate model. **Accumulated over the first ~10 s of speech** (not wall clock —
  silence does not count), confidence-weighted, then **locked** for the session
  unless the user overrides (D32; see PRD, user inputs). Locking on the first
  utterance alone is a coin flip when that utterance is "okay, hi".
- **A transcript is rejected** — never published — when the utterance is shorter
  than `min_utterance_ms`, `no_speech_prob` is above threshold, or `avg_logprob`
  is below threshold (D30). A fabricated caption is worse than a missing one:
  the user has no way to tell it is fabricated.
- `asr_ms` is not diagnostics-only: `asr_ms / (end_ms - start_ms)` is the live
  real-time factor and it is the honest answer to "is this actually real-time".

---

## 4. SentenceSplitter

Turns transcripts into **caption-sized units**. The caption unit is the
**sentence**, not the utterance (D29).

```python
@dataclass
class Sentence:
    id: str                    # sentence-scoped: "u_0042.s1"
    utterance_id: str
    text: str
    language: str
    preceding_silence_ms: int  # only on the FIRST sentence of an utterance; 0 after
    closed_by: Literal["silence", "max_length"]   # from the parent Utterance
    is_flush: bool             # emitted by the flush rule, not by a sentence end
```

`closed_by` and `preceding_silence_ms` are carried through from `Utterance`
because the caption message needs them and `Translation` does not carry them —
see the join note under Publisher.

**Rule:**
1. Prepend any held fragment from the previous utterance to `Transcript.text`.
2. Split on sentence boundaries.
3. Emit every **complete** sentence immediately.
4. **Hold** the trailing fragment for the next utterance.

Complete sentences are not delayed at all, so this buys translation quality
without spending any of the latency budget (D25).

**The flush rule is not optional.** A held fragment must be emitted when silence
exceeds `2 × silence_threshold_ms`, on `stop`, or at session end. Without it the
last words spoken in a meeting are held forever and never appear.

---

## 5. Translator

```python
@dataclass
class Translation:
    sentence_id: str
    source_text: str
    target_text: str | None    # None when translation failed — UI shows source only
    source_lang: str
    target_lang: str
    mt_ms: int
    mt_error: str | None
```

Google Cloud Translation, free tier (D7), called over **REST with an API key**
rather than the client library (D37) — one `httpx` POST to
`translation.googleapis.com/language/translate/v2`, key from `.env`. Keep this
behind an interface: the provider is the most likely thing to change, and the
free-tier character budget is finite.

**Consumed by exactly one serialized task (D27).** Transcripts leave the worker
in FIFO order, but concurrent translations can complete out of order and would
scramble the caption stream. One task consuming an ordered queue costs no
measurable latency — utterance rate is far below translation throughput — and
removes the need for a reorder buffer.

**Required behaviours (D31):**
- **Persist** cumulative characters to a file keyed by month. 500 000/month, no
  roll-over. Warn at 80%, hard-stop at a configurable ceiling. An in-memory
  counter cannot see the realistic failure, which is a testing loop across many
  process restarts.
- **LRU cache** on `(source_lang, target_lang, text)`. Meeting speech repeats
  ("okay", "yes", "can you hear me") — free characters and free latency.
- **Skip entirely when `source_lang == target_lang`.** Saves the round trip and
  the characters.
- **Never block the pipeline on failure.** Retry with backoff, then publish with
  `target_text = None` and `mt_error` set.

---

## 6. Publisher → UI (WebSocket, JSON)

**FastAPI + uvicorn**, one process serving both the static UI and the WebSocket
(D38). The UI is plain HTML/CSS/JS — no npm, no bundler, no build step.

The Publisher **joins** each `Translation` back to its `Sentence` by
`sentence_id`. `Translation` deliberately carries only what the translator
produced; the caption's `utterance_id`, `preceding_silence_ms` and `closed_by`
come from the `Sentence` side of that join.

Server → browser messages:

```jsonc
// A finished caption. One SENTENCE, not one utterance (D29).
{ "type": "caption",
  "id": "u_0042.s1",                  // sentence-scoped
  "utterance_id": "u_0042",
  "source_text": "¿Me escuchan bien?",
  "target_text": "Can you hear me okay?",   // null if translation failed (D31)
  "mt_error": null,
  "source_lang": "es",
  "target_lang": "en",
  "preceding_silence_ms": 1240,       // first sentence of the utterance only; 0 after
  "closed_by": "silence",
  "latency_ms": { "asr": 0, "mt": 0, "word_age": 0 } }

// Live speaking indicator — high frequency, no persistence
{ "type": "vad", "speaking": true }

// Detected source language, awaiting confirmation or override
{ "type": "language_detected", "lang": "es", "confidence": 0.98 }

// Session state. Drives the UI's six defined states — see PRD.
{ "type": "state",
  "state": "loading",                 // idle|loading|listening|running|degraded|error
  "detail": "Loading model (first run downloads weights)" }

// Utterances discarded under backpressure (D28). The gap must be visible.
{ "type": "dropped", "after_id": "u_0041.s2", "utterances": 1 }

// Honest real-time health. Surface this in the UI.
{ "type": "health",
  "queue_depth": 0,
  "rtf": 0.41,
  "word_age_ms": 6100,                // the D25 number, measured — not RTF
  "effective_max_utterance_ms": 4000, // shrinks under load (D28); shown so
                                      // adaptation is observable, not mysterious
  "chars_used": 12840,
  "chars_budget": 500000 }
```

Browser → server messages:

```jsonc
{ "type": "set_target_lang", "lang": "en" }
{ "type": "override_source_lang", "lang": "pt" }   // locks it for the session
{ "type": "start" }
{ "type": "stop" }
```

### On the `health` message

`queue_depth` is the depth of the `Segmenter → Transcriber` queue. If it stays
near zero you are keeping up; if it grows, RTF has gone above 1 and the app is
falling progressively further behind.

`word_age_ms` is the one the DoD actually asks about. **RTF is a stability
metric, not a latency metric** (D25): it says the queue will not diverge, not
that the captions are close to the speaker. A configuration can hold RTF at 0.4
and still put the first word of a caption twelve seconds behind. Publish both.

```
word_age = silence_threshold + MT_roundtrip + D × (1 + RTF)
```

These cost almost nothing to expose and together they are the honest answer to
the DoD's real-time requirement. A graph of them across a ten-minute session is
evidence for the journey doc rather than a claim — see D34 for the JSONL log
that produces it.
