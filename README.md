# Automatisk-sentimentanalys

**Svenskt Call Center Intelligence-system** med sentimentanalys, ASR (tal-till-text), speaker diarization, intent-klassificering och LLM-stöd (Mistral/OpenRouter).

GDPR-vänligt, skalbart och byggt för svenska kundtjänstsamtal.

## Snabbstart

```bash
pip install -e ".[cli,api,dashboard-nicegui]"

# CLI
sentimentanalys --help

# API
uvicorn src.api:app --reload

# Dashboard
python -m app.nicegui_dashboard.main
```

Se `docs/` och `ROADMAP.md` för mer information.