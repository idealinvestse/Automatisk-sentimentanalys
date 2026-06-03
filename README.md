# Automatisk sentimentanalys (Flashback)

Ett minimalt, körbart system för svensk sentimentanalys med Hugging Face Transformers. Första versionen fokuserar på offline-analys (ingen scraping) och ett CLI som tar text, .txt eller .csv och producerar etikett (negativ, neutral, positiv) + sannolikhet.

## Funktioner (v0-v1)
- Enkel CLI: analysera enstaka text, en .txt med en text per rad, eller en .csv med vald kolumn
- Svensk-kapabel modell: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multispråk, fungerar bra för forum/tweet-liknande text)
- Output till terminal eller CSV med kolumner: text, label, score, model, timestamp
- (Ny) Valfritt svenskt lexikon för blending
- (Ny) ASR (tal-till-text) för telefonsamtal: KBLab `kb-whisper-large` och OpenAI `whisper-large-v3`
  - CLI: `src/cli.py` med kommandon `sentiment`, `transcribe` och `analyze-call`
  - REST API: `/transcribe`, `/analyze_conversation`, `/batch_transcribe`, `/batch_analyze_conversation`, `/scan_process`
- (Ny) Utvärderingsramverk: `src/evaluate.py` för att mäta prestanda mot testset

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
- **Caching & cost**: Inbyggd innehålls-adresserbar disk-cache (.cache/llm/). Upprepade körningar på samma transkript ≈ 0 kr.
- **Output**: `report.llm` + `results["llm"]` (additivt). Dashboard visar "LLM-enhanced" badge + dedikerade expanders för actionable insights, agent assessment, trajectory etc.
- **Utvärdering**: `python -m src.evaluate llm-quality` ger proxy-metrics (fallback rate, cost, evidence coverage, consistency).
- **Nya moduler**: `src/llm/{openrouter_client.py, mistral_analyzer.py, prompts.py, schemas.py}`
- **Viktigt**: Kräver `OPENROUTER_API_KEY`. Utan nyckel faller allt tillbaka till lokal analys (aldrig krasch).

**Dokumentation för nya användare**:
- Praktisk quickstart + privacy + exempel: `docs/FAS3_MISTRAL_LLM_INTEGRATION.md` (mål: igång på <30 min)
- Full plan, tasks & status: `UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md`
- Arkitektur: `docs/ARCHITECTURE.md` (uppdaterad med LLM-lager)

Se även `configs/llm_config.yaml` (exempel) och `reports/llm_quality_smoke.json` (efter evaluate).

## Installation (Windows)

**Rekommenderat:** installer, portable ZIP eller dev-setup — se **[docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md)**.

```powershell
# Utvecklare (git clone)
.\scripts\dev-setup.ps1 -Profile cli -InitConfig
.\launcher.ps1 doctor
Sentimentanalys.bat              # GUI launcher (dubbelklick / Start-meny)
streamlit run app/setup_hub.py   # konfiguration
```

Manuell venv (samma som tidigare):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-min.txt -r requirements-cli.txt -r requirements-install.txt
# API: även -r requirements-api.txt
# Desktop: även -r requirements-desktop.txt
```

Notera (ASR): ffmpeg krävs för `--preprocess` (portable/installer kan bunda under `tools\ffmpeg\bin`).

## Körningsexempel
```powershell
# Enstaka text
python -m src.cli sentiment --text "Vaccin är säkert och viktigt för folkhälsan."

# Från .txt (en text per rad) och spara till CSV
python -m src.cli sentiment --txt-file samples\examples.txt --output outputs\predictions.csv

# Från .csv med kolumnnamn
python -m src.cli sentiment --csv-file path\till\data.csv --text-column kommentar --output outputs\preds.csv

# Välj annan modell (om du vill experimentera)
python -m src.cli sentiment --text "Det här är dåligt" --model cardiffnlp/twitter-xlm-roberta-base-sentiment

# Fulla klass-sannolikheter (negativ/neutral/positiv) + auto-enhet
python -m src.cli sentiment --text "Det här var otroligt bra!" --return-all-scores --device auto

# Batch med fulla sannolikheter och kortare maxlängd
python -m src.cli sentiment --txt-file samples\examples.txt --return-all-scores --max-length 256 --output outputs\predictions_all.csv

# Profiler (automatisk anpassning efter datatyp/källa)
# Exempel: forum-innehåll med rensning av användarnamn/hashtags
python -m src.cli sentiment --txt-file samples\examples.txt --source forum --return-all-scores --output outputs\predictions_forum.csv

# Exempel: magasin/artikel med längre maxlängd
python -m src.cli sentiment --text "Lång artikeltext..." --datatype article --return-all-scores

# Lexikon (valfritt / auto): blanda modell med svenskt lexikon
# För forum/callcenter etc. auto-används nu data/sensaldo_lexicon.csv (vikt från profil) om inte explicit anges.
# Överstyr eller ange för andra profiler:
# Prova med samples/lexicon_sample.csv och 30% lexikonvikt
python -m src.cli sentiment --txt-file samples\examples.txt --source forum \
  --return-all-scores --lexicon-file samples\lexicon_sample.csv --lexicon-weight 0.3 \
  --output outputs\predictions_lex.csv

# REST API-anrop med lexikon
curl -X POST \
  http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Det här var fantastiskt!",
      "Riktigt dålig service."
    ],
    "source": "forum",
    "datatype": "post",
    "return_all_scores": true,
    "lexicon_file": "samples/lexicon_sample.csv",
    "lexicon_weight": 0.3
  }'

### ASR: CLI
```powershell
# Transkribera ljudfil (kb-whisper-large med faster-whisper, strict revision för call center)
python -m src.cli transcribe path\till\call.wav --backend faster --language sv --revision strict --output-json outputs\call_transcript.json

# Alternativ backend (Transformers)
python -m src.cli transcribe path\till\call.wav --backend transformers --model openai/whisper-large-v3

# Analysera samtal per segment (sentiment + ev. lexikonblending)
python -m src.cli analyze-call path\till\call.wav \
  --backend faster --language sv \
  --lexicon-file samples\lexicon_sample.csv --lexicon-weight 0.3 \
  --output-csv outputs\call_segments.csv

# Avancerade ASR-flaggor (hotwords, initial_prompt, preprocess, chunking)
python -m src.cli analyze-call path\till\call.wav \
  --backend faster --revision strict \
  --hotwords "fakturering,återbetalning,kundtjänst" \
  --initial-prompt "Detta är ett svenskt kundsamtal om faktura och support." \
  --preprocess \
  --chunk-length-s 30

# Batch-transkribering: mappar, listor och globbar
# 1) Katalog (rekursivt)
python -m src.cli transcribe data\calls --backend faster --language sv --output-dir outputs\transcripts --log-level INFO

# 2) Glob-mönster (rekursivt)
python -m src.cli transcribe "data\\calls\\**\\*.wav" --backend faster --output-dir outputs\transcripts

# 3) Flera filer i en körning
python -m src.cli transcribe data\a.wav data\b.mp3 data\c.m4a --output-dir outputs\transcripts

# Batch-analys av samtal med aggregerad CSV över alla filer
python -m src.cli analyze-call "data\\calls\\**\\*.wav" \
  --backend faster --language sv \
  --lexicon-file samples\lexicon_sample.csv --lexicon-weight 0.25 \
  --output-csv outputs\all_call_segments.csv --log-level INFO

# Loggning
# Lägg till --log-level DEBUG för mer detaljerad logg (modell, enhet, segment mm.)
python -m src.cli transcribe path\till\call.wav --log-level DEBUG
```

### ASR: REST API
```bash
# Transkribera
curl -X POST http://localhost:8000/transcribe -H "Content-Type: application/json" -d '{
  "audio_path": "samples/call.wav",
  "model": "kb-whisper-large",
  "backend": "faster",
  "language": "sv"
}'

# Analysera konversation (transkribera + sentiment per segment)
curl -X POST http://localhost:8000/analyze_conversation -H "Content-Type: application/json" -d '{
  "audio_path": "samples/call.wav",
  "backend": "faster",
  "language": "sv",
  "return_all_scores": true,
  "lexicon_file": "samples/lexicon_sample.csv",
  "lexicon_weight": 0.25
}'
```

#### ASR: REST API - Batch & Scan

```bash
# Batch: transkribera flera filer (filer/mappar/globbar) med parallellism
curl -X POST http://localhost:8000/batch_transcribe -H "Content-Type: application/json" -d '{
  "audio_paths": ["data/a.wav", "data/b.mp3"],
  "directory": "data/calls",
  "glob": "**/*.wav",
  "recursive": true,
  "limit": 50,
  "workers": 2,
  "model": "kb-whisper-large",
  "backend": "faster",
  "device": "auto",
  "language": "sv",
  "beam_size": 5,
  "vad": true,
  "word_timestamps": true
}'

# Batch: transkribera + analysera samtal per segment
curl -X POST http://localhost:8000/batch_analyze_conversation -H "Content-Type: application/json" -d '{
  "directory": "data/calls",
  "glob": "**/*.wav",
  "workers": 2,
  "model": "kb-whisper-large",
  "backend": "faster",
  "language": "sv",
  "word_timestamps": false,
  "sentiment_model": null,
  "lexicon_file": "samples/lexicon_sample.csv",
  "lexicon_weight": 0.25
}'

# Skanna katalog och processa nya/uppdaterade filer i små batcher (inkrementellt)
# - Håller koll via state_file (JSON) på senaste mtime per fil
# - operation: "transcribe" eller "analyze_conversation"
curl -X POST http://localhost:8000/scan_process -H "Content-Type: application/json" -d '{
  "directory": "incoming/calls",
  "pattern": "**/*.wav",
  "recursive": true,
  "batch_size": 4,
  "workers": 2,
  "max_files": 100,
  "state_file": "state/scan_state.json",
  "operation": "transcribe",
  "model": "kb-whisper-large",
  "backend": "faster",
  "language": "sv",
  "lexicon_file": null,
  "lexicon_weight": 0.0
}'
```

Notera:
- **workers**: trådar per batch (1-8) för parallell körning.
- **state_file**: JSON som spårar `processed` filer med `mtime`; endast nya/ändrade filer körs.
- **batch_size**: antal filer per batch; endpointen kör batchar sekventiellt men kan parallellisera inom batch.
- **glob/pattern**: använder Python glob (t.ex. `**/*.wav`).

#### Full Pipeline API

```bash
# Kör sentiment + intent + topics + insights + risks på befintliga segments
curl -X POST http://localhost:8000/analyze_pipeline -H "Content-Type: application/json" -d '{
  "segments": [
    {"text": "Jag är mycket missnöjd med fakturan", "speaker": "kund"},
    {"text": "Jag ska hjälpa dig direkt", "speaker": "agent"}
  ],
  "device": "cpu"
}'
```

#### Python: End-to-end Pipeline

```python
from src.pipeline import CallAnalysisPipeline

pipe = CallAnalysisPipeline(profile="callcenter")

# Från audio (inkl. transkribering + diarization + all analys)
# Nya flaggor (Fas 1-2): hotwords, initial_prompt, preprocess, aspects/emotions/trajectory etc. via registry
report = pipe.analyze_audio(
    "data/call.wav",
    num_speakers=2,
    language="sv",
    hotwords=["fakturering", "support"],
    preprocess=True,
)

# Resultat innehåller nu även:
# report.results["aspect"], ["emotion"], ["role"], ["trajectory"], ...
print(report.to_dict())
```

#### Dashboard

```bash
# Starta Streamlit-dashboardet
streamlit run app/dashboard.py
```

Dashboardet visar:
- Sentiment-trender över tid
- Intent-fördelning (pie chart)
- Agent-prestanda (resolution rate, sentiment)
- "Hot topics" från utvärdering
- Live-analys av egna segments

## Utvärdering

Projektet inkluderar ett utvärderingsramverk för att mäta sentimentanalysens prestanda:

```powershell
# Kör baseline-utvärdering med default-profil
python -m src.evaluate evaluate --testset data/test_swedish.csv --output reports/baseline.json

# Utvärdera med specifik profil (callcenter/forum etc. auto-använder nu lexicon från profil-default)
python -m src.evaluate evaluate --profile callcenter
# Överstyr explicit vid behov:
# python -m src.evaluate evaluate --profile call --lexicon-file samples/lexicon_sample.csv --lexicon-weight 0.3

# Spara detaljerade resultat som CSV
python -m src.evaluate evaluate --output-csv reports/detailed_results.csv

# Lista tillgängliga profiler
python -m src.evaluate list-profiles
```

### Benchmark (baseline)

`reports/baseline_results.json` innehåller tre obligatoriska scenarier:

| Scenario | Profil | Backend | Accuracy | Macro-F1 |
|---|---|---|---:|---:|
| Forum | `forum` | heuristic baseline | 49.52% | 47.07% |
| Call | `call` | heuristic baseline | 49.52% | 47.07% |
| News | `news` | heuristic baseline | 49.52% | 47.07% |

Kör skarp Hugging Face-utvärdering med:

```powershell
python -m src.evaluate evaluate --backend model --profile call --output reports/call_model_results.json
```

## KB-Whisper: Rekommenderad ASR-modell

**KBLab** (KB, National Library of Sweden) har släppt svensktränade Whisper-modeller
med ~47% lägre WER (Word Error Rate) jämfört med OpenAI:s modeller på svenska.

Tillgängliga revisioner för `KBLab/kb-whisper-large`:

| Revision   | Beskrivning | Rekommenderas för |
|-----------|------------|-------------------|
| `standard` | Standard-transkribering | Generellt bruk |
| `strict`   | Verbatim (ordagrant) - behåller utfyllnadsord, upprepningar | **Call center (rekommenderas)** |
| `subtitle` | Bättre läsbarhet, skiljetecken, versaler | Undertexter, visning |

```powershell
# Använd KB-Whisper med strict-revision (rekommenderas för call center)
python -m src.asr_cli transcribe samtal.wav --model kb-whisper-large --revision strict

# Använd subtitle-revision för bättre läsbarhet
python -m src.asr_cli transcribe samtal.wav --revision subtitle

# Byt tillbaka till OpenAI Whisper-baseline
python -m src.asr_cli transcribe samtal.wav --model openai/whisper-large-v3
```

## Fas 1–3: Avancerad Call Center Intelligence (slutfört per UTVECKLINGSPLAN.md)

Se `UTVECKLINGSPLAN.md` för detaljerad task-lista (1.1–3.4).

Huvudsakliga leveranser (utöver tidigare):

- **ASR-förbättringar** (1.1–1.4): WhisperX-backend, hotwords/initial_prompt (med auto-laddning av `configs/callcenter_hotwords.txt`), chunking+overlap+low_confidence i faster, valbar `--preprocess` (ffmpeg high-pass + noisereduce).
- **ABSA** (1.5): `aspect`-analyzer (hybrid keywords + befintlig sentiment). 8 callcenter-aspects. Finns i `results["aspect"]`.
- **Fas 2-analyzers** (via `src/analysis/` registry):
  - `emotion`: multi-label (frustration, ilska, oro m.fl.)
  - `role`: agent/customer inference
  - `trajectory`: escalation detection + sentiment slope
  - `spoken_normalizer`: tar bort "eh", "hmm" etc.
  - `llm_judge`: stub för low-conf fall (utbyggbar)
- **Pipeline & API**: Alla nya analyzers körs automatiskt (eller via `selected_analyzers`). `report.results` är additivt.
- **LoRA/Finetune** (3.1): Befintlig pipeline i `src/finetune.py` + `configs/finetune.yaml` + callcenter-data.

Rekommenderad call center-konfiguration:

- ASR: `KBLab/kb-whisper-large` med `--revision strict` (eller `whisperx` för bättre alignment)
- `--hotwords`, `--preprocess`, `--profile callcenter`
- Lexikon + LearnedBlender + per-segment low-conf boost
- Nya vyer i dashboard (aspects, emotions, trajectory) rekommenderas att byggas ut.

```powershell
# Exempel med alla nya flaggor
python -m src.cli analyze-call call.wav \
  --backend faster --revision strict \
  --hotwords "fakturering,återbetalning" \
  --preprocess \
  --profile callcenter \
  --lexicon-weight 0.25
```

## Utveckling

```powershell
# Installera utvecklingsberoenden
pip install pytest pytest-cov ruff black mypy

# Kör tester
pytest tests/ -v

# Kör tester med coverage
pytest tests/ -v --cov=src --cov-report=term

# Linting
python -m ruff check src/
python -m black --check src/

# Formattering
python -m black src/
```

## Minimal modul-användning
Vill du använda systemet som en liten modul i egen kod:

```python
from src.sentiment import analyze, load

# 1) Snabb enradare (cachad pipeline under huven)
texts = [
    "Det här var fantastiskt!",
    "Riktigt dålig service.",
]
results = analyze(texts)  # [{'label': 'positiv', 'score': ...}, ...]
for r in results:
    print(r)

# 2) Egen instans + annan modell (valfritt)
# Auto-enhet (cuda/mps/cpu), fulla sannolikheter
sp = load("cardiffnlp/twitter-xlm-roberta-base-sentiment", device="auto", return_all_scores=True, max_length=256)
print(sp.analyze(["Vet inte riktigt... "]))

# 3) Profil-medveten analys (rensning, modellval, auto lexicon från profil)
from src.sentiment import analyze_smart
results, meta = analyze_smart(
    ["Det här är fantastiskt!"],
    profile="callcenter",  # auto: lexicon (data/sensaldo_lexicon.csv @0.25) + heuristics + intensity + emoji etc.
    return_all_scores=True,
    # lexicon_file/weight kan anges för override; nya opt: map_emojis etc via profile cleaning
)
print(results, meta)
# Meta innehåller nu ev. "lexicon_file", "lexicon_weight", "hybrid_lexicon_boost" etc.
```

### ASR: Minimal användning
```python
from src.transcription import get_transcriber

# Transkribera med KB-Whisper large (strict revision) + hotwords + preprocess
transcriber = get_transcriber(
    backend="faster",
    model_name="kb-whisper-large",
)
result = transcriber.transcribe(
    audio_path="samtal.wav",
    language="sv",
    revision="strict",
    hotwords=["fakturering", "återbetalning", "kundtjänst"],
    preprocess=True,  # high-pass + optional noise reduction
).to_dict()
print(result["segments"])
for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    if seg.get("properties", {}).get("low_confidence"):
        print("  (low confidence – högre lexicon-vikt användes i analys)")
```

## Projektstruktur
```
src/sentiment.py           # Sentimentanalys (Hugging Face)
src/analysis/              # Registry-baserade analyzers (sentiment, aspect, emotion, trajectory, role, ...)
src/cli.py                 # Huvud-CLI (sentiment, transkribering, samtalsanalys)
src/transcription/         # ASR backends (faster, transformers, whisperx) + preprocess.py
src/lexicon.py             # Lexikon-baserad sentiment + blending (LearnedBlender)
src/clean.py               # Textrensning
src/profiles.py            # Profilhantering (inkl. callcenter-aspects)
src/evaluate.py            # Utvärderingsramverk
src/api/                   # FastAPI REST-server
src/pipeline.py            # End-to-end CallAnalysisPipeline
src/core/                  # Modeller, errors, config, audio
tests/                     # Enhetstester
data/                      # Testdata
docs/                      # Dokumentation
configs/                   # callcenter_hotwords.txt m.m.
samples/                   # Exempelfiler
```

## Licens
MIT - se LICENSE för detaljer.