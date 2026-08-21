"""Hallucination guard (D30).

Whisper emits confident garbage on near-silence — "Thank you.", subtitle-credit
boilerplate.  Silero cuts most of these but a notification chime or a 400 ms
cough still gets through and still produces a caption.  A fabricated caption is
worse than a missing one: the user cannot tell it is fabricated (D30).
"""

from __future__ import annotations

from speech_translator import config as cfg


def is_accepted(
    no_speech_prob: float,
    avg_logprob: float,
    config: cfg.Config,
) -> bool:
    """Return True when the utterance passes the hallucination guard.

    Reject if Whisper itself says the input is likely not speech, OR if the
    average token log-probability is too low (low-confidence output).
    """
    return (
        no_speech_prob <= config.no_speech_prob_max
        and avg_logprob >= config.avg_logprob_min
    )
