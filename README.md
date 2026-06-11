# Automatisk sentimentanalys (Flashback)

Ett **svenskt Call Center Intelligence-system** med sentimentanalys, ASR (tal-till-text), speaker diarization, intent-klassificering, aspect-based sentiment, emotion detection, trajectory-analys och valfri Mistral/OpenRouter LLM-förstärkning.

> **Mål**: Ge call center och kundtjänst företag ett kraftfullt, GDPR-vänligt verktyg för automatisk analys av telefonsamtal på svenska.

## Quickstart & Installation (rekommenderat)

### 1. Grundinstallation (CLI + Sentiment + ASR)

```bash
git clone https://github.com/idealinvestse/Automatisk-sentimentanalys.git
cd Automatisk-sentimentanalys

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[cli]"
```

### 2. Full Call Center + Diarization (rekommenderas för analyze-call)

```bash
pip install -e ".[cli,diarize]"
```

> **Viktigt**: `pyannote.audio` kräver ofta en Hugging Face token (`huggingface-cli login` eller `HF_TOKEN` env). Se [pyannote documentation](https://huggingface.co/pyannote/speaker-diarization-3.1) för modellåtkomst.

### 3. REST API + Dashboard

```bash
pip install -e ".[api]"
# Starta API
uvicorn src.api.server:app --reload
# Starta Streamlit dashboard
streamlit run app/dashboard.py
```

### Docker (rekommenderas för produktion/test)

```bash
docker build -t automatisk-sentimentanalys .
docker run --rm -it --gpus all \
  -v $(pwd)/samples:/app/samples \
  -v $(pwd)/outputs:/app/outputs \
  automatisk-sentimentanalys analyze-call samples/call.wav --backend faster --language sv
```

### Windows-användare (enkelt)

Se **[docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md)** för portable ZIP, `Sentimentanalys.bat` och `dev-setup.ps1`.

### Systemkrav (ffmpeg)

För `--preprocess` (Task 1.4) krävs `ffmpeg` installerat på systemet:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Ladda ner från https://ffmpeg.org eller använd portable/installer-paketet.

## Hardware Requirements

| Användningsfall              | Rekommenderad HW                  | Kommentar |
|-------------------------------|-----------------------------------|---------|
| Enkel sentiment (text/csv)    | CPU, 4 GB RAM                     | Snabbt på bärbar |
| ASR (faster-whisper large)    | GPU 8+ GB VRAM, 16 GB RAM         | Starkt rekommenderat |
| + Speaker Diarization         | GPU 10–12+ GB VRAM             | pyannote.audio är tung |
| Full pipeline + LLM (OpenRouter) | GPU 8+ GB VRAM                 | Lokala modeller + extern LLM |
| API / Batch processing        | 16+ GB RAM, bra CPU/GPU           | Flera workers |

**Tips**: Använd `--device auto` eller `device="cuda"` / `"cpu"` för att styra.

## Funktioner (v0-v1)
- Enkel CLI: analysera enstaka text, en .txt med en text per rad, eller en .csv med vald kolumn
- Svensk-kapabel modell: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multispråk, fungerar bra för forum/tweet-liknande text)
- Output till terminal eller CSV med kolumner: text, label, score, model, timestamp
- Valfritt svenskt lexikon för blending
- ASR (tal-till-text) för telefonsamtal: KBLab `kb-whisper-large` och OpenAI `whisper-large-v3`
  - CLI: `src/cli.py` med kommandon `sentiment`, `transcribe` och `analyze-call`
  - REST API: `/transcribe`, `/analyze_conversation`, `/batch_transcribe`, `/batch_analyze_conversation`, `/scan_process` — se **[docs/API.md](docs/API.md)** (v0.4.0, auth, Fas 4, exempel)
- Utvärderingsramverk: `src/evaluate.py` för att mäta prestanda mot testset

## Funktioner (v0.3 – Call Center Intelligence)
- **Speaker Diarization**: Separera agent och kund med pyannote.audio eller energy-based VAD
- **Intent Classification**: 10 call center-intents med keyword- och model-backend
- **Call Summarization**: Extractive summary, action items, och outcome-detection
- **Topic Modeling**: Keyword-baserad topic extraction med sentiment-fördelning
- **Root Cause Analysis**: Identifiera upprepade klagomål, faktureringsproblem, tekniska fel
- **Predictive Analytics**: Churn-risk, escalation-risk, och satisfaction scoring
- **End-to-end Pipeline**: `CallAnalysisPipeline` som binder samman alla moduler
- **Streamlit Dashboard**: Visuell översikt över sentiment, intents, topics och agent-prestanda
- **Full Pipeline API**: `/analyze_pipeline` – kör alla analyssteg i ett anrop

### Avancerad Call Center Intelligence (Fas 1–3 – UTVECKLINGSPLAN.md)
- **ASR Backends**: `faster` (default, KB-Whisper), `transformers`, `whisperx` (bättre alignment + inbyggd diarization)
- **Hotwords & Initial Prompt** (Task 1.3): `--hotwords "fakturering,återbetalning"` + `--initial-prompt`. Auto-laddar `configs/callcenter_hotwords.txt` för callcenter.
- **Chunking + Confidence** (Task 1.2): 30s chunks + 5s overlap i faster-backend för långa filer. `low_confidence` flag + automatisk högre lexicon-vikt på osäkra segment.
- **Pre-processing** (Task 1.4): `--preprocess` – ffmpeg high-pass filter + valfri noisereduce före ASR (bättre WER på bullriga inspelningar).
- **Aspect-Based Sentiment (ABSA)** (Task 1.5): `aspect` analyzer (hybrid keyword + sentiment). Callcenter-aspects: fakturering_pris, kundtjänst_kvalitet, teknisk_lösning, väntetid, agent_attityd m.fl. Finns i `results["aspect"]`.
- **Emotions** (Fas 2.1): Multi-label emotion detection (frustration, ilska, besvikelse, oro, glädje m.fl.).
- **Role Inference** (Fas 2.2): Heuristisk agent vs customer-klassificering baserat på talmönster.
- **Trajectory & Escalation** (Fas 2.3): Sentiment/emotion-tidsserie, slope, escalation_events.
- **Spoken Normalizer** (Fas 2.5): Tar bort fillers ("eh", "hmm") efter strict-transkription.
- **LLM-Judge stub** (Fas 2.4): Plats för low-confidence fall (för närvarande heuristic fallback).
- Alla nya analyzers är registrerade i `src/analysis/` (topologisk körning, error isolation) och aktiveras automatiskt eller via `selected_analyzers`.
- `CallAnalysisReport.results` innehåller nu `aspect`, `emotion`, `role`, `trajectory` m.m. (additivt, backward compatible).

### Mistral / OpenRouter LLM Integration (Fas 3 – European-first holistisk analys)
- **Hybrid-arkitektur**: Lokala modeller + heuristics är default/fast path. Mistral (via OpenRouter) används selektivt för **full-conversation** reasoning: trajectory/escalation, root cause, actionable QA-rekommendationer, agent assessment med evidensspann.
- **Primärmodell**: `mistralai/mistral-medium-3.5` (stark svensk prestanda, 256k context). Starkare alternativ `mistralai/mistral-large-3`.
- **Strict structured output**: `response_format` med `json_schema` + `strict: true` → garanterat parsebar Pydantic-validerad JSON.
- **Aktivering**:
  - Per profil: `callcenter` har `llm.enabled: true` (selektiv via längd/låg confidence).
  - Explicit: `analyze-call ... --use-mistral-llm --llm-model mistralai/mistral-medium-3.5`
  - API: `use_mistral_llm`, `llm_model`, `deep_analysis` i PipelineRequest.
  - Pipeline: `CallAnalysisPipeline(use_mistral_llm=True)`
- **Privacy / GDPR**: Varje extern anrop loggar tydligt "EXTERNAL LLM CALL (OpenRouter/Mistral) ... data sent to third-party". Caching gör om-körningar gratis. Cost tracking i meta. Budget-varning stöd i klienten.
- **Caching & cost**: Inbyggd innehålls-adresserbar disk-cache (`.cache/llm/`). Upprepade körningar på samma transkript ≈ 0 kr.
- **Output**: `report.llm` + `results["llm"]` (additivt). Dashboard visar "LLM-enhanced" badge + dedikerade expanders för actionable insights, agent assessment, trajectory etc.
- **Utvärdering**: `python -m src.evaluate llm-quality` ger proxy-metrics (fallback rate, cost, evidence coverage, consistency).
- **Nya moduler**: `src/llm/{openrouter_client.py, mistral_analyzer.py, prompts.py, schemas.py}`
- **Viktigt**: Kräver `OPENROUTER_API_KEY`. Utan nyckel faller allt tillbaka till lokal analys (aldrig krasch).

**Dokumentation för nya användare**:
- Praktisk quickstart + privacy + exempel: `docs/FAS3_MISTRAL_LLM_INTEGRATION.md` (mål: igång på <30 min)
- Full plan, tasks & status: `UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md`
- Arkitektur: `docs/ARCHITECTURE.md` (uppdaterad med LLM-lager)

Se även `configs/llm_config.yaml` (exempel) och `reports/llm_quality_smoke.json` (efter evaluate).

## Installation (Windows) - Legacy / Avancerad

För Windows-användare som föredrar portable-installer eller `dev-setup.ps1`:

**Rekommenderat:** Se **[docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md)**.

```powershell
.\scripts\dev-setup.ps1 -Profile cli -InitConfig
.\launcher.ps1 doctor
Sentimentanalys.bat              # GUI launcher
streamlit run app/setup_hub.py   # konfiguration
```

Manuell venv (fortfarande fungerande):

```powershell
python -m venv .venv
.\ .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-min.txt -r requirements-cli.txt
# För diarization: pip install -e ".[diarize]" (efter ovan)
```

Notera: `ffmpeg` krävs för `--preprocess` (kan buntas i portable-paket).