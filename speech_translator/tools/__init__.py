"""Operator commands, run as ``python -m speech_translator.tools.<name>``.

These are not part of the pipeline. They exist so the audio layer can be
inspected and captured from on the demo machine without starting the server —
and because `docs/BENCHMARK.md` sources its sample audio through
`record_loopback`, which puts one of them on the critical path for Level 3.
"""
