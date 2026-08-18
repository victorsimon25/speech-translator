# PRD — Real-Time Speech Translator

## The brief (as supplied)

> Here are the outputs:
> - Translated text in target language
> - Detect silences and showcase punctuation (Bonus)
>
> Definition of Done:
> - Design a good UX and decide on the appropriate input to take from the user
> - Have a UI that you can demonstrate
>   - UI should be clear for the user without any additional explanations
> - Speech Translator should work in real-time (the latency should be minimal).
> - It should be able to translate speech between **[MISSING — TRUNCATED]**
>
> - This is not a test to check your existing knowledge and capabilities. We
>   want to see how you use LLMs to overcome the limitations of your
>   **[MISSING — TRUNCATED]**
> - Use a Cursor or LLM Powered IDE or LLMs like a very smart pair programmer.
>   Don't just instruct; seek advice, review, and guide it.
> - Create a journey doc and share it with us at least 48 hours before the
>   interview. Capture a lot of screenshots to document your journey. This is
>   **[MISSING — TRUNCATED]**
> - Be sure to capture your prompts.
> - Document your assessment of the LLM output.
> - Record your learning and recovery from situations where the LLM didn't meet
>   expectations.
> - Utilize LLMs extensively. From the initial idea, coding, testing, LLMs can
>   assist throughout the entire development process.
> - Focus on the most important functionality
> - Don't limit yourself with the stack, technology.
> - The open-ended nature of these tasks is intentional, encouraging you to
>   think critically and creatively. There is no single right answer, so trust
>   your instincts to guide you.

### ⚠ Blockers — must be resolved before implementation

1. **The scope line is truncated.** *"It should be able to translate speech
   between …"* — this determines the required number of languages and whether
   one-directional (incoming-only) translation satisfies the brief at all. The
   current design assumes incoming-only; if the line reads "between two people"
   or similar, that assumption is wrong and the design changes.
2. **The deadline is unknown.** The journey doc is due 48 hours before the
   interview, so the effective deadline is earlier than the interview date.
3. **The bonus requirement is ambiguous.** *"Detect silences and showcase
   punctuation"* could mean (a) output text should be correctly punctuated —
   which Whisper provides natively, so it is effectively already done — or
   (b) the UI should visibly show pauses between utterances, which is real UI
   work. The `Utterance` contract carries `preceding_silence_ms` so that (b)
   remains cheap if that is the intent.

## Product summary

The user is in an online meeting (Zoom, Teams, Meet, Slack huddle — any of
them). Someone speaks a language they don't understand. This app captures the
meeting audio from the operating system, transcribes it, translates it into a
language the user chose, and displays running captions in a browser window
beside the meeting.

## Scope

**In scope**
- Capture system audio (the mixed output of everything playing on the machine).
- Detect the source language automatically, show it, allow the user to override
  it, then lock it for the session.
- Transcribe speech to punctuated text.
- Translate into a user-selected target language (3–5 supported).
- Display captions in real time with minimal latency.
- Run on Linux and Windows, tested on both.

**Out of scope**
- Multi-user / multi-tenant scaling. Explicitly excluded by the user.
- Text-to-speech. Output is text only.
- Outgoing (microphone) translation. The input layer is pluggable so this stays
  a small addition, but it is not being built.
- Per-platform meeting integrations (Zoom Apps, Teams bots). See `DECISIONS.md`
  for the evidence behind rejecting these.

**Stretch, only if time and CPU allow**
- Anonymous speaker diarization ("Speaker 1 / Speaker 2"). Real names are not
  obtainable from loopback audio — see `DECISIONS.md`.
- Desktop shell around the browser UI.

## Definition of done

- [ ] Captures system audio on Linux **and** Windows, verified on both.
- [ ] Source language auto-detected, displayed, and overridable.
- [ ] Target language selectable from a fixed set.
- [ ] Captions appear with minimal latency and do not visibly rewrite themselves.
- [ ] Output text is correctly punctuated.
- [ ] The UI needs no explanation to use.
- [ ] Sustained real-time factor stays below 1.0 for the length of a demo — the
      app must not fall progressively further behind.
- [ ] Journey doc with prompts, screenshots, and assessment of LLM output.

## User inputs

The DoD asks us to "decide on the appropriate input to take from the user". The
answer is deliberately small:

| Input | Required? | Why |
|---|---|---|
| Target language | **Yes** | Cannot be inferred. The one genuinely necessary input. |
| Audio device | Once | Defaults to the system output monitor; only shown if there is more than one candidate. |
| Source language | **No — suggested** | Auto-detected and displayed as a correctable suggestion, not a question. Locked after confirmation. |

The design principle: ask for exactly one thing, infer the rest, and make every
inference visible and correctable.
