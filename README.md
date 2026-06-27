# Automatisk-sentimentanalys

**Svenskt Call Center Intelligence-system** med sentimentanalys, ASR (tal-till-text), speaker diarization, intent-klassificering och LLM-stöd (Mistral/OpenRouter/Groq).

GDPR-vänligt, skalbart och byggt för svenska kundtjänstsamtal.

> **För absolut senaste systembeskrivning, feature status och komplett agent-kontext:**  
> Se [PROJECT_STATUS.md](PROJECT_STATUS.md) och [AGENT_CONTEXT.md](AGENT_CONTEXT.md).  
> Dessa filer hålls synkade med kodbasen via github-project-status skill.  
> **Last updated:** 2026-06-27

## Grok Build / AI Agent Development (snabbstart)

Detta repo är optimerat för Grok Build och andra LLM coding agents.

```bash
# Setup
pip install -e ".[cli,api,dashboard-nicegui]"
sentimentanalys download-asr

# Verifiera
pytest --tb=no -q

# Starta dashboard
python -m app.nicegui_dashboard.main
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

# Dashboard
python -m app.nicegui_dashboard.main
```

### Snabb transkribering (dashboard)

Fliken **Transkribering** har en sektion **Snabb transkribering** högst upp: ladda upp en ljudfil (`.wav`, `.mp3`, `.m4a`, …), klicka **Transkribera nu** och se segment med tidsstämplar, confidence och talare direkt i UI. Exportera till JSON/CSV eller kopiera hela transkriptet.

Vid **Använd Backend API**: sätt `API_MEDIA_ROOT` till projektroten så uppladdade filer kan nås via `POST /transcribe`.

### Windows-launcher (ASR)

```powershell
.\launcher.ps1 asr-status          # visa paket + modellcache
.\launcher.ps1 asr-install         # installera faster-whisper + whisperx
.\launcher.ps1 asr-download        # förladda modeller
.\launcher.ps1 provision           # full install inkl. ASR (eller GUI: Hantera ASR / Transkribering)
```

Se `docs/`, `ROADMAP.md`, `PROJECT_STATUS.md` och `AGENT_CONTEXT.md` för mer information.

---

**Note on sections about setup, quickstart, or AI/agents:**

- **Setup/Quickstart:** The "Snabbstart" section provides installation commands (`pip install`, model downloads, CLI/API/Dashboard startup) and Windows launcher scripts for ASR setup.
- **AI/Agents:** The system includes AI components such as sentiment analysis, ASR (speech-to-text), speaker diarization, intent classification, and LLM support (Mistral/OpenRouter/Groq). References to `AGENT_CONTEXT.md` suggest agent-related functionality, though details are not in this file.