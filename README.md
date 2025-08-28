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

# 2) Uppgradera pip och installera beroenden
python -m pip install -U pip
pip install -r requirements.txt
```

## Körningsexempel
```powershell
# Enstaka text
python -m src.main run --text "Vaccin är säkert och viktigt för folkhälsan."

# Från .txt (en text per rad) och spara till CSV
python -m src.main run --txt-file samples\examples.txt --output outputs\predictions.csv

# Från .csv med kolumnnamn
python -m src.main run --csv-file path\till\data.csv --text-column kommentar --output outputs\preds.csv

# Välj annan modell (om du vill experimentera)
python -m src.main run --text "Det här är dåligt" --model cardiffnlp/twitter-xlm-roberta-base-sentiment
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
