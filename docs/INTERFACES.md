# Interfaces

The contract between stages. **A session working on one stage should be able to
read only this file plus that stage's own source.** Keep it that way — if you
find yourself reading another stage's implementation to understand its output,
this file is wrong and should be fixed.

## Pipeline

```
AudioSource → Segmenter → Transcriber → Translator → Publisher → UI
 (platform)     (VAD +      (Whisper,     (Google      (WebSocket)  (browser)
                max-len)     local CPU)    Cloud)
             └──────────┘  └────── separate process (GIL) ──────┘
```

Every arrow is a queue. The `Segmenter → Transcriber` queue crosses a **process**
boundary (see D15) — the rest may be threads or coroutines.

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
- Fixed-size chunks. Size is dictated by the VAD: Silero wants 512 samples
  (32 ms); WebRTC VAD wants 10/20/30 ms. Pick the VAD first, then fix the frame
  size to match and do not vary it.

**Implementations:**

| Class | Platform | Backing |
|---|---|---|
| `PipeWireMonitorSource` | Linux | PipeWire monitor of the default sink — verified available (D2) |
| `WasapiLoopbackSource` | Windows | `PyAudioWPatch` WASAPI loopback (D3) |
| `WavFileSource` | any | Reads a WAV at wall-clock speed. **Use this for tests** — deterministic, no live audio needed |
| `MicrophoneSource` | future | Only if two-way is required (D4) |

Source selection happens once at startup and is the *only* place platform
detection is allowed to appear.

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

**Tunables** (these are the two numbers you will spend tuning time on):
- `silence_threshold_ms` — start around 500–800
- `max_utterance_ms` — start around 6000–8000

`preceding_silence_ms` is carried so the silence-visualisation bonus is cheap if
that turns out to be what the brief means. It costs nothing to populate.

The Segmenter also emits a continuous boolean `speaking` signal for the live UI
indicator (D11). That is a side channel, not part of `Utterance`.

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
```

- Model size is decided by `BENCHMARK.md`. Do not hard-code one before that runs.
- `language` is populated from Whisper's built-in language identification — no
  separate model. Detected on the first utterance, surfaced to the UI, then
  **locked** for the session unless the user overrides (see PRD, user inputs).
- `asr_ms` is not diagnostics-only: `asr_ms / (end_ms - start_ms)` is the live
  real-time factor and it is the honest answer to "is this actually real-time".

---

## 4. Translator

```python
@dataclass
class Translation:
    utterance_id: str
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    mt_ms: int
```

Google Cloud Translation, free tier (D7). Keep this behind an interface — the
provider is the most likely thing to change, and the free-tier character budget
is finite.

**Track cumulative characters sent.** The free tier is 500 000/month and does not
roll over. Burning it in testing before demo day would be an avoidable failure.

---

## 5. Publisher → UI (WebSocket, JSON)

Server → browser messages:

```jsonc
// A finished caption
{ "type": "caption",
  "id": "u_0042",
  "source_text": "¿Me escuchan bien?",
  "target_text": "Can you hear me okay?",
  "source_lang": "es",
  "target_lang": "en",
  "preceding_silence_ms": 1240,
  "closed_by": "silence",
  "latency_ms": { "asr": 0, "mt": 0, "total": 0 } }

// Live speaking indicator — high frequency, no persistence
{ "type": "vad", "speaking": true }

// Detected source language, awaiting confirmation or override
{ "type": "language_detected", "lang": "es", "confidence": 0.98 }

// Honest real-time health. Surface this in the UI.
{ "type": "health", "queue_depth": 0, "rtf": 0.41 }
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

It costs almost nothing to expose and it is the single honest metric for the
DoD's real-time requirement. A graph of it across a ten-minute session is
evidence for the journey doc rather than a claim.
