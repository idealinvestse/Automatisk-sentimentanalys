# Automatisk sentimentanalys (Flashback)

Ett minimalt, körbart system för svensk sentimentanalys med Hugging Face Transformers. Första versionen fokuserar på offline-analys (ingen scraping) och ett CLI som tar text, .txt eller .csv och producerar etikett (negativ, neutral, positiv) + sannolikhet.

## Funktioner (v0)
- Enkel CLI: analysera enstaka text, en .txt med en text per rad, eller en .csv med vald kolumn
- Svensk-kapabel modell: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multispråk, fungerar bra för forum/tweet-liknande text)
- Output till terminal eller CSV med kolumner: text, label, score, model, timestamp

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

# (Kompatibilitet) Allt-i-ett
# pip install -r requirements.txt
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

# 3) Direkt med parametrar via hjälpfunktionen
probs = analyze(["Toppen!", "Uselt..."], device="auto", return_all_scores=True, max_length=256)
print(probs[0])
```

## Profiler och datatyper
- __Tillgängliga profiler__: `default`, `forum`, `magazine`, `news`, `social`, `review`.
- __Automatiska val__:
  - Modell: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (standard i alla profiler nu)
  - `max_length`: 256 för forum/social/review, 512 för magazine/news
  - Rensning per profil: ta bort URL:er, ev. HTML, användarnamn, hashtags, normalisera whitespace, m.m.
- __Mapping__ (förenklad):
  - `--source forum|flashback|reddit` -> `forum`
  - `--source magazine` -> `magazine`
  - `--source news|newspaper` -> `news`
  - `--source twitter|x|social|blog` -> `social`
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
    "return_all_scores": true
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
- `requirements.txt` – Python-beroenden
- `samples/examples.txt` – Exempeltexter på svenska
- `outputs/` – Skapas automatiskt vid export (git-ignorerad)

## Noteringar om kvalitet och etik
- Den valda modellen är tränad för sociala medier och fungerar ofta bra även för forum. För högsta kvalitet krävs ev. finslipning/träning på Flashback-lik data.
- Denna v0 skrapar inte data. Om du vill skrapa Flashback, säkerställ att juridiska/etiska aspekter (GDPR, ToS, robots.txt) hanteras korrekt.
- Första körningen laddar ner modellen från Hugging Face (~hundratals MB), vilket kan ta några minuter.

## Nästa steg
- (Valfritt) Lägg till enkel utvärdering på etiketterad testdata
- (Valfritt) Streamlit-gränssnitt
- (Valfritt) Integrera svenskt lexikon (t.ex. SenSALDO) som komplement
