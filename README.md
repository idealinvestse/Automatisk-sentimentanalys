# Automatisk sentimentanalys (Flashback)

Ett minimalt, körbart system för svensk sentimentanalys med Hugging Face Transformers. Första versionen fokuserar på offline-analys (ingen scraping) och ett CLI som tar text, .txt eller .csv och producerar etikett (negativ, neutral, positiv) + sannolikhet.

## Funktioner (v0-v1)
- Enkel CLI: analysera enstaka text, en .txt med en text per rad, eller en .csv med vald kolumn
- Svensk-kapabel modell: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multispråk, fungerar bra för forum/tweet-liknande text)
- Output till terminal eller CSV med kolumner: text, label, score, model, timestamp
- (Ny) Valfritt svenskt lexikon för blending
- (Ny) ASR (tal-till-text) för telefonsamtal: KBLab `kb-whisper-large` och OpenAI `whisper-large-v3`
  - CLI: `src/asr_cli.py` med kommandon `transcribe` och `analyze-call`
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

## Installation (Windows PowerShell)
```powershell
# 1) Skapa och aktivera virtuell miljö
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Uppgradera pip
python -m pip install -U pip

# Alternativ A: Minimal modul (endast inferens)
pip install -r requirements-min.txt

# Alternativ B: CLI (modul + kommandoradsverktyg)
pip install -r requirements-min.txt -r requirements-cli.txt

# (Valfritt) API (FastAPI) - inkluderar ASR-beroenden
pip install -r requirements-min.txt -r requirements-api.txt

# (Kompatibilitet) Allt-i-ett
# pip install -r requirements.txt

# Notera (ASR): För bästa prestanda, installera ffmpeg i systemet (Dockerfilen hanterar detta).
```

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

# Lexikon (valfritt): blanda modell med svenskt lexikon
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

pipe = CallAnalysisPipeline()

# Från audio (inkl. transkribering + diarization + all analys)
report = pipe.analyze_audio("data/call.wav", num_speakers=2, language="sv")

# Eller från befintliga segments
segments = [
    {"text": "Jag har problem med min faktura", "start": 0, "end": 5},
    {"text": "Jag ska kolla på det direkt", "start": 5, "end": 10},
]
report = pipe.analyze_segments(segments)

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

# Utvärdera med specifik profil och lexikon-blending
python -m src.evaluate evaluate --profile call --lexicon-file samples/lexicon_sample.csv --lexicon-weight 0.3

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

## Fas 2: Domänanpassning

Fas 2 introducerar en reproducerbar LoRA/PEFT-pipeline för svensk call center-domän.

```powershell
# Installera träningsberoenden
pip install -r requirements.txt

# Kör LoRA fine-tuning (kräver GPU för praktisk körning)
python -m src.finetune --config configs/finetune.yaml

# Utvärdera ny adapter/model output
python -m src.evaluate evaluate --backend model --model models/callcenter-sentiment-lora --profile callcenter
```

Rekommenderad call center-konfiguration:

- ASR: `KBLab/kb-whisper-large` med `--revision strict`
- Sentimentprofil: `callcenter`
- Lexikon: `data/sensaldo_lexicon.csv`
- Lexikon-blending: börja med `--lexicon-weight 0.25`

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

# 3) Profil-medveten analys (rensning, modellval)
from src.sentiment import analyze_smart
results, meta = analyze_smart(
    ["Det här är fantastiskt!"],
    profile="call",
    return_all_scores=True,
)
print(results, meta)
```

### ASR: Minimal användning
```python
from src.asr import transcribe

# Transkribera med KB-Whisper large (strict revision)
result = transcribe(
    audio_path="samtal.wav",
    model="kb-whisper-large",
    backend="faster",
    language="sv",
    revision="strict",
)
print(result["segments"])
for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
```

## Projektstruktur
```
src/sentiment.py      # Sentimentanalys (Hugging Face)
src/asr.py            # ASR (tal-till-text) med faster-whisper/transformers
src/lexicon.py        # Lexikon-baserad sentiment
src/clean.py          # Textrensning
src/profiles.py       # Profilhantering
src/evaluate.py       # Utvärderingsramverk
src/main.py           # Huvud-CLI för sentiment
src/asr_cli.py        # CLI för ASR
src/api.py            # FastAPI REST-server
tests/                # Enhetstester
data/                 # Testdata
docs/                 # Dokumentation
samples/              # Exempelfiler
```

## Licens
MIT - se LICENSE för detaljer.