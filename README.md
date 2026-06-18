# Automatisk-sentimentanalys

**Svenskt Call Center Intelligence-system** med sentimentanalys, ASR (tal-till-text), speaker diarization, intent-klassificering och LLM-stöd (Mistral/OpenRouter).

GDPR-vänligt, skalbart och byggt för svenska kundtjänstsamtal.

## Ny funktion: YouTube Data Ingestion (Fas 4–6)

Ladda ner ljud från YouTube för att bygga testdataset.

### CLI
```bash
sentimentanalys download-youtube "https://www.youtube.com/watch?v=..." [--playlist] [--no-wav]
```

### REST API
```bash
POST /ingest/youtube/download
{
  "url": "...",
  "playlist": false,
  "convert_to_wav": true,
  "auto_transcribe": true,
  "auto_analyze": false
}
```

### NiceGUI Dashboard
Ny flik **"Datainsamling"** med:
- URL + inställningar
- Progress + status
- Lista över filer med Transkribera / Analysera / Ta bort

### Hjälpverktyg
```bash
python scripts/ingest_youtube_data.py   # Förbereder JSONL för testset
```

**Viktigt:** Använd endast offentligt material. Alla nedladdningar loggas med källa.

Se full plan: `docs/INTEGRATION_PLAN_YOUTUBE_DOWNLOADER.md`

## Snabbstart

```bash
pip install -e ".[cli,api,dashboard-nicegui,data]"

# CLI
sentimentanalys --help

# API
uvicorn src.api:app --reload

# Dashboard
python -m app.nicegui_dashboard.main
```

Se `docs/` och `ROADMAP.md` för mer information.