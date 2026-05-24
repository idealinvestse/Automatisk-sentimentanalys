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
python -m src.main --text "Vaccin är säkert och viktigt för folkhälsan."

# Från .txt (en text per rad) och spara till CSV
python -m src.main --txt-file samples\examples.txt --output outputs\predictions.csv

# Från .csv med kolumnnamn
python -m src.main --csv-file path\till\data.csv --text-column kommentar --output outputs\preds.csv

# Välj annan modell (om du vill experimentera)
python -m src.main --text "Det här är dåligt" --model cardiffnlp/twitter-xlm-roberta-base-sentiment

# Fulla klass-sannolikheter (negativ/neutral/positiv) + auto-enhet
python -m src.main --text "Det här var otroligt bra!" --return-all-scores --device auto

# Batch med fulla sannolikheter och kortare maxlängd
python -m src.main --txt-file samples\examples.txt --return-all-scores --max-length 256 --output outputs\predictions_all.csv

# Profiler (automatisk anpassning efter datatyp/källa)
# Exempel: forum-innehåll med rensning av användarnamn/hashtags
python -m src.main --txt-file samples\examples.txt --source forum --return-all-scores --output outputs\predictions_forum.csv

# Exempel: magasin/artikel med längre maxlängd
python -m src.main --text "Lång artikeltext..." --datatype article --return-all-scores

# Lexikon (valfritt): blanda modell med svenskt lexikon
# Prova med samples/lexicon_sample.csv och 30% lexikonvikt
python -m src.main --txt-file samples\examples.txt --source forum \
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
python -m src.asr_cli transcribe path\till\call.wav --backend faster --language sv --revision strict --output-json outputs\call_transcript.json

# Alternativ backend (Transformers)
python -m src.asr_cli transcribe path\till\call.wav --backend transformers --model openai/whisper-large-v3

# Analysera samtal per segment (sentiment + ev. lexikonblending)
python -m src.asr_cli analyze-call path\till\call.wav \
  --backend faster --language sv \
  --lexicon-file samples\lexicon_sample.csv --lexicon-weight 0.3 \
  --output-csv outputs\call_segments.csv

# Batch-transkribering: mappar, listor och globbar
# 1) Katalog (rekursivt)
python -m src.asr_cli transcribe data\calls --backend faster --language sv --output-dir outputs\transcripts --log-level INFO

# 2) Glob-mönster (rekursivt)
python -m src.asr_cli transcribe "data\\calls\\**\\*.wav" --backend faster --output-dir outputs\transcripts

# 3) Flera filer i en körning
python -m src.asr_cli transcribe data\a.wav data\b.mp3 data\c.m4a --output-dir outputs\transcripts

# Batch-analys av samtal med aggregerad CSV över alla filer
python -m src.asr_cli analyze-call "data\\calls\\**\\*.wav" \
  --backend faster --language sv \
  --lexicon-file samples\lexicon_sample.csv --lexicon-weight 0.25 \
  --output-csv outputs\all_call_segments.csv --log-level INFO

# Loggning
# Lägg till --log-level DEBUG för mer detaljerad logg (modell, enhet, segment mm.)
python -m src.asr_cli transcribe path\till\call.wav --log-level DEBUG
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