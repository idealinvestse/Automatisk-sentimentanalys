# Edge AI — On-device Swedish Call Center Analysis (MVP)

**Version:** v0.5 (EDGE-01)

## Scope

Offline analysis without external LLM or network calls:

- Sentiment (local model + lexicon)
- Intent (heuristic)
- Early PII redaction for `callcenter` profile
- Optional ASR via local faster-whisper (CLI `--audio`)

**Not included:** LLM holistic analysis, pyannote diarization, Fas 4 aggregate API.

## CLI

```bash
sentimentanalys edge-analyze --text "Tack för hjälpen, det fungerade bra!"
sentimentanalys edge-analyze --audio samples/audio/sv/demo.wav --profile callcenter
```

## Contract

Output type: `EdgeAnalysisResult` in `src/edge/contracts.py`

```python
{
  "profile": "callcenter",
  "offline": true,
  "llm_used": false,
  "segments": [...],
  "limitations": ["No LLM", "No diarization (pyannote)", ...]
}
```

## Roadmap

- ONNX export for sentiment model
- Quantized ASR bundle for edge devices
- Sync with `src/llm/routing.py` for hybrid edge/cloud handoff

See archived [SYNERGY_ANALYSIS_Fas5.md](archive/SYNERGY_ANALYSIS_Fas5.md) for long-term vision.
