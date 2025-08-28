# Automatisk sentimentanalys (Flashback)

Ett minimalt, körbart system för svensk sentimentanalys med Hugging Face Transformers. Första versionen fokuserar på offline-analys (ingen scraping) och ett CLI som tar text, .txt eller .csv och producerar etikett (negativ, neutral, positiv) + sannolikhet.

## Funktioner (v0–v1)
- Enkel CLI: analysera enstaka text, en .txt med en text per rad, eller en .csv med vald kolumn
- Svensk-kapabel modell: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multispråk, fungerar bra för forum/tweet-liknande text)
- Output till terminal eller CSV med kolumner: text, label, score, model, timestamp
- (Ny) Valfritt svenskt lexikon för blending
- (Ny) ASR (tal-till-text) för telefonsamtal: KBLab `kb-whisper-large` och OpenAI `whisper-large-v3`
  - CLI: `src/asr_cli.py` med kommandon `transcribe` och `analyze-call`
  - REST API: `/transcribe`, `/analyze_conversation`, `/batch_transcribe`, `/batch_analyze_conversation`, `/scan_process`

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

# (Valfritt) API (FastAPI) – inkluderar ASR-beroenden
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
# Transkribera ljudfil (kb-whisper-large med faster-whisper)
python -m src.asr_cli transcribe path\till\call.wav --backend faster --language sv --output-json outputs\call_transcript.json

# Alternativ backend (Transformers)
python -m src.asr_cli transcribe path\till\call.wav --backend transformers --model openai/whisper-large-v3

# Transkribera utan sentimentanalys (endast transkribering)
python -m src.asr_cli transcribe path\till\call.wav --backend faster --language sv --mode transcribe-only --output-json outputs\call_transcript.json

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

# Batch-transkribering utan sentimentanalys
python -m src.asr_cli transcribe data\calls --backend faster --language sv --mode transcribe-only --output-dir outputs\transcripts

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

#### ASR: REST API – Batch & Scan

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
- __workers__: trådar per batch (1–8) för parallell körning.
- __state_file__: JSON som spårar `processed` filer med `mtime`; endast nya/ändrade filer körs.
- __batch_size__: antal filer per batch; endpointen kör batchar sekventiellt men kan parallellisera inom batch.
- __glob/pattern__: använder Python glob (t.ex. `**/*.wav`).


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

# 3) Direkt med parametrar via hjälpfunktionen
probs = analyze(["Toppen!", "Uselt..."], device="auto", return_all_scores=True, max_length=256)
print(probs[0])
```

## Profiler och datatyper
- __Tillgängliga profiler__: `default`, `forum`, `magazine`, `news`, `social`, `review`, `call`.
- __Automatiska val__:
  - Modell: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (standard i alla profiler nu)
  - `max_length`: 256 för forum/social/review, 512 för magazine/news
  - Rensning per profil: ta bort URL:er, ev. HTML, användarnamn, hashtags, normalisera whitespace, m.m.
- __Mapping__ (förenklad):
  - `--source forum|flashback|reddit` -> `forum`
  - `--source magazine` -> `magazine`
  - `--source news|newspaper` -> `news`
  - `--source twitter|x|social|blog` -> `social`
  - `--source call|phone|telephony` -> `call`
  - `--datatype post|comment` -> `forum`, `--datatype article|story` -> `news`, `--datatype review` -> `review`
- __Åsidosättningar__: du kan alltid sätta `--profile forum` eller välja `--model` och `--max-length` manuellt.

## REST API
En lättvikts-API byggd med FastAPI finns i `src/api.py`.

- __Start lokalt__ (utanför Docker):
  ```powershell
  pip install -r requirements-min.txt -r requirements-api.txt
  uvicorn src.api:app --host 0.0.0.0 --port 8000
  ```
  Öppna http://localhost:8000/docs för Swagger UI.

- __Request-exempel (JSON)__:
  ```json
  {
    "texts": [
      "Det här var fantastiskt!",
      "Riktigt dålig service."
    ],
    "source": "forum",
    "datatype": "post",
    "return_all_scores": true,
    "lexicon_file": "samples/lexicon_sample.csv",
    "lexicon_weight": 0.3
  }
  ```

## Docker
Bygg och kör allt inuti en container.

```powershell
# Bygg image
docker build -t sv-sentiment .

# Skapa cache-volym för modeller (snabbare start efter första körning)
docker volume create hf_cache

# Kör API på port 8000
docker run --rm -p 8000:8000 -v hf_cache:/cache/hf sv-sentiment

# Öppna Swagger: http://localhost:8000/docs

# (Valfritt) Kör CLI inuti container och spara resultat till volym
docker volume create sentiment_outputs
docker run --rm -v hf_cache:/cache/hf -v sentiment_outputs:/app/outputs sv-sentiment \
  python -m src.main --txt-file samples/examples.txt --source forum --return-all-scores --output outputs/preds.csv
```

## Filstruktur
- `src/main.py` – CLI och inferens
- `src/asr.py` – ASR (Whisper) med faster-whisper/Transformers
- `src/asr_cli.py` – CLI för transkribering och samtalsanalys
- `requirements.txt` – Python-beroenden
- `samples/examples.txt` – Exempeltexter på svenska
- `outputs/` – Skapas automatiskt vid export (git-ignorerad)


## Nästa steg
- (Valfritt) Lägg till enkel utvärdering på etiketterad testdata
- (Valfritt) Streamlit-gränssnitt
- (Valfritt) Integrera svenskt lexikon (t.ex. SenSALDO) som komplement
