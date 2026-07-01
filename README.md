# Automatisk-sentimentanalys

**Svenskt Call Center Intelligence-system** med sentimentanalys, ASR (tal-till-text), speaker diarization, intent-klassificering och LLM-stöd (Mistral/OpenRouter/Groq).

GDPR-vänligt, skalbart och byggt för svenska kundtjänstsamtal.

> **Status och roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md) · [PROJECT_STATUS.md](PROJECT_STATUS.md)  
> **För agenter:** [AGENT_CONTEXT.md](AGENT_CONTEXT.md) → [docs/LLM_AGENT_GUIDE.md](docs/LLM_AGENT_GUIDE.md)  
> **Last updated:** 2026-06-28

## Grok Build / AI Agent Development (snabbstart)

Detta repo är optimerat för Grok Build och andra LLM coding agents.

```bash
# Setup
pip install -e ".[cli,api,dashboard-nicegui]"
sentimentanalys download-asr

# Verifiera
pytest --tb=no -q

# Starta backend-API
uvicorn src.api:app --reload

# Starta web UI (Next.js, primär dashboard)
cd webui && npm install && npm run dev   # → http://localhost:3000

# Legacy NiceGUI-dashboard (fortfarande tillgänglig)
python -m app.archive.nicegui_dashboard.main
```

Läs [AGENTS.md](AGENTS.md) först – den pekar till `docs/LLM_AGENT_GUIDE.md` som är den kompletta guiden för agenter.

## Snabbstart

```bash
pip install -e "[cli,api,dashboard-nicegui]"

# Hämta faster-whisper, whisperx och standardmodeller (kb-whisper-large m.fl.)
sentimentanalys download-asr
# eller: sentimentanalys-download-asr

# CLI
sentimentanalys --help

# API
uvicorn src.api:app --reload

# Web UI (primär dashboard – Next.js + React + Tailwind)
cd webui && npm install && npm run dev   # → http://localhost:3000

# Legacy NiceGUI-dashboard
python -m app.archive.nicegui_dashboard.main
```

> **Frontend-status:** `webui/` (Next.js) är den primära frontenden från Fas 4.
> `app/archive/nicegui_dashboard/` är kvar som legacy men rekommenderas inte för nytt
> arbete. Se [docs/WEBUI_MODERNIZATION_PLAN.md](docs/WEBUI_MODERNIZATION_PLAN.md)
> för migreringsstatus. Docker: `docker compose -f docker-compose.webui.yml up --build`.

### Snabb transkribering (web UI)

Fliken **Transkribering** i web UI visar live-loggar och jobbstatus från
backendens WebSocket (`/ws/transcription`). För ad-hoc pipeline-tester, använd
fliken **Testlabb** som kör `/analyze_pipeline` direkt på JSON-segment.

### Windows-launcher (ASR)

```powershell
.\launcher.ps1 asr-status          # visa paket + modellcache
.\launcher.ps1 asr-install         # installera faster-whisper + whisperx
.\launcher.ps1 asr-download        # förladda modeller
.\launcher.ps1 provision           # full install inkl. ASR (eller GUI: Hantera ASR / Transkribering)
```

Se [docs/](docs/), [docs/ROADMAP.md](docs/ROADMAP.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) och [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) för mer information.
