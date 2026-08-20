# Translation API Migration Note

**Date**: 2026-08-19 (session 9, during Level 3 benchmark)

## Context

During benchmark execution, Google Cloud Translation API was not used because it requires a billing account to be enabled (even for free-tier usage). The benchmark proceeded without measuring `MT_roundtrip`, which means:
- Gate 3 (p95 word age) reported as `unknown` for all configurations
- Model selection relied on gates 1 (RTF) and 2 (VRAM) only
- Word age predictions use the assumed 300 ms MT latency from D25/D51

## Consideration: LibreTranslate

**LibreTranslate** offers a public API that could replace Google Translate:
- Open-source translation engine
- Free public API available at `https://libretranslate.com/`
- Self-hostable for production use
- REST API similar to Google's

**Migration would require**:
1. Update `speech_translator/config.py`:
   - `translate_url` → LibreTranslate endpoint
   - Remove `google_translate_api_key` requirement
   - Adjust request/response format (similar but not identical)
2. Update translation module to match LibreTranslate API contract
3. Re-measure `MT_roundtrip` latency (likely higher than Google's CDN)
4. Update `docs/DECISIONS.md` with D37 supersession

**Status**: Under consideration. Not blocking Level 3 completion since model selection can proceed on RTF + VRAM alone. If implemented, re-run gate 3 evaluation with measured MT latency.

## Impact on Level 3 Benchmark

- **Gates 1 & 2**: Unaffected (RTF and VRAM are measured independently of translation)
- **Gate 3**: Reported as `unknown` in results table
- **Model selection**: Still valid based on "fastest that is accurate enough" rule (D25/D26)
- **Level 7**: End-to-end word age measurement will include actual MT latency once API is integrated

This note serves as a marker for future sessions. The benchmark results remain valid for comparing model performance.
