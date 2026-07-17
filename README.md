# Automatisk-sentimentanalys

**Svenskt Call Center Intelligence-system** med sentimentanalys, ASR (tal-till-text), speaker diarization, intent-klassificering och LLM-stöd (Mistral/OpenRouter/Groq).

GDPR-vänligt, skalbart och byggt för svenska kundtjänstsamtal.

> **Status och roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md)  
> **För agenter:** [AGENTS.md](AGENTS.md) → [docs/LLM_AGENT_GUIDE.md](docs/LLM_AGENT_GUIDE.md)  
> **Last updated:** 2026-07-13

## Snabbstart

```bash
pip install -e ".[cli,api,dev]"   # inkludera [dev] för pytest

# Hämta faster-whisper, whisperx och standardmodeller (kb-whisper-large m.fl.)
sentimentanalys download-asr

# Verifiera (kräver [dev] / scripts/dev-setup.ps1 -Profile dev)
pytest --tb=no -q

# CLI
sentimentanalys --help

# API
uvicorn src.api:app --reload

# Web UI (Next.js)
cd webui && npm install && npm run dev   # → http://localhost:3000
```

Läs [AGENTS.md](AGENTS.md) först om du är en coding agent – den pekar till `docs/LLM_AGENT_GUIDE.md`.

> **Frontend:** `webui/` (Next.js) är den enda dashboarden.  
> Docker: `docker compose -f docker-compose.webui.yml up --build`.

### Snabb transkribering (web UI)

Fliken **Transkribering** i web UI visar live-loggar och jobbstatus från
backendens WebSocket (`/ws/transcription`). För ad-hoc pipeline-tester, använd
fliken **Testlabb** som kör `/analyze_pipeline` direkt på JSON-segment.

### Windows-launcher (ASR)

```powershell
.\launcher.ps1 asr-status          # visa paket + modellcache
.\launcher.ps1 asr-install         # installera faster-whisper + whisperx
.\launcher.ps1 asr-download        # förladda modeller
.\launcher.ps1 provision           # full install inkl. ASR
```

Launcher synkar API-port, `NEXT_PUBLIC_API_BASE_URL`, API-nyckel och CORS till webui
vid start — se [docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md).

**Första fem WAV-testet (Windows + GPU):** [docs/FIRST_FIVE_WAV_TEST.md](docs/FIRST_FIVE_WAV_TEST.md) — placera fem demo-`.wav` i `samples/audio/sv/callcenter/`, kör `.\scripts\run_first_five_wav_test.ps1`.

Se [docs/](docs/) och [docs/ROADMAP.md](docs/ROADMAP.md) för mer information.
