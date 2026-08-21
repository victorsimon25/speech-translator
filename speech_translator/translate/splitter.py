"""Split transcripts into caption-sized sentences (INTERFACES.md §4, D29)."""

from __future__ import annotations

import re
from typing import Literal

from ..transcribe.protocol import Transcript
from .protocol import Sentence


class SentenceSplitter:
    """Turn Transcript objects into Sentence objects, carrying fragments across utterances.

    Complete sentences are emitted immediately; the trailing fragment is held and
    prepended to the next utterance's text before splitting again (D29).
    """

    def __init__(self) -> None:
        self._fragment: str = ""
        self._fragment_language: str = ""
        self._fragment_utterance_id: str | None = None
        self._sentence_counters: dict[str, int] = {}

    def feed(
        self,
        transcript: Transcript,
        preceding_silence_ms: int = 0,
        closed_by: Literal["silence", "max_length"] = "silence",
    ) -> list[Sentence]:
        """Prepend held fragment, split on sentence boundaries, return complete sentences.

        preceding_silence_ms and closed_by are not stored on Transcript (they live on
        the parent Utterance); the main pipeline passes them as keyword arguments.
        """
        uid = transcript.utterance_id
        if uid not in self._sentence_counters:
            self._sentence_counters[uid] = 0

        if self._fragment:
            full_text = (self._fragment + " " + transcript.text).strip()
        else:
            full_text = transcript.text.strip()

        # Split after .?! followed by whitespace, keeping punctuation with its sentence.
        parts = re.split(r"(?<=[.?!])\s+", full_text)

        # The last part is a fragment unless it ends with sentence-closing punctuation.
        if parts and not parts[-1].endswith((".", "?", "!")):
            new_fragment = parts.pop()
        else:
            new_fragment = ""

        sentences: list[Sentence] = []
        first = True
        for part in parts:
            part = part.strip()
            if not part:
                continue
            self._sentence_counters[uid] += 1
            n = self._sentence_counters[uid]
            sentences.append(
                Sentence(
                    id=f"{uid}.s{n}",
                    utterance_id=uid,
                    text=part,
                    language=transcript.language,
                    preceding_silence_ms=preceding_silence_ms if first else 0,
                    closed_by=closed_by,
                    is_flush=False,
                )
            )
            first = False

        self._fragment = new_fragment
        self._fragment_language = transcript.language
        self._fragment_utterance_id = uid

        return sentences

    def flush(
        self,
        closed_by: Literal["silence", "max_length"],
        preceding_silence_ms: int,
    ) -> Sentence | None:
        """Emit the held fragment as a flush sentence; return None if nothing is held."""
        if not self._fragment:
            return None

        uid = self._fragment_utterance_id
        assert uid is not None
        if uid not in self._sentence_counters:
            self._sentence_counters[uid] = 0
        self._sentence_counters[uid] += 1
        n = self._sentence_counters[uid]

        sentence = Sentence(
            id=f"{uid}.s{n}",
            utterance_id=uid,
            text=self._fragment,
            language=self._fragment_language,
            preceding_silence_ms=preceding_silence_ms,
            closed_by=closed_by,
            is_flush=True,
        )
        self._fragment = ""
        self._fragment_utterance_id = None
        self._fragment_language = ""
        return sentence
