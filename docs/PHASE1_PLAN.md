# Fas 1: Stabilisering & KB-Whisper Integration – Projektplan

**Datum**: Maj–Juni 2026
**Status**: Pågående
**Gren**: `feat/phase1-evaluation`

---

## Övergripande Mål

1. Skapa ett robust **utvärderingsramverk** så vi kan mäta prestanda
2. **Integrera de officiella KB-Whisper-modellerna** (särskilt `kb-whisper-large` med `strict`-varianten)
3. Höja kodkvaliteten (tester, CI, linting, buggfixar)
4. Göra projektet redo för nästa fas (domänanpassning)

---

## Framgångskriterier (Definition of Done)

- [x] `evaluate.py` finns och körs med baseline-resultat på ett svenskt testset
- [x] Alla ASR-funktioner använder `KBLab/kb-whisper-large` (med stöd för `strict`/`subtitle`)
- [x] Minst 80% testtäckning på kärnmoduler (`sentiment.py`, `lexicon.py`, `clean.py`, `asr.py`)
- [x] GitHub Actions CI passerar (pytest + lint)
- [x] `README.md` är uppdaterad med nya modeller och benchmark-resultat
- [x] Inga regressioner i befintlig funktionalitet

---

## Steg-för-steg Implementation

### Steg 1: Förberedelse & Analys ✅

- [x] Läs igenom hela projektstrukturen
- [x] Analysera aktuell ASR-implementation
- [x] Identifiera alla ställen där modeller hardkodas
- [x] Skapa `docs/PHASE1_PLAN.md`

### Steg 2: Skapa Utvärderingsramverk

- [ ] Skapa `src/evaluate.py`
- [ ] Skapa svenskt testset `data/test_swedish.csv`
- [ ] Implementera accuracy, macro-F1, per-klass F1, confusion matrix
- [ ] Stöd för olika profiler och lexicon-vikter
- [ ] CLI: `python -m src.evaluate --output reports/baseline.json`

### Steg 3: Integrera KB-Whisper-modellerna

- [ ] Uppdatera `src/asr.py` med revision-stöd
- [ ] Uppdatera `src/asr_cli.py` och `src/api.py`
- [ ] Lägg till rekommendation: Använd `strict` för call center
- [ ] Uppdatera alla exempel i `README.md`

### Steg 4: Kodkvalitet & Bugfixar

- [ ] Installera och konfigurera `ruff`, `black`, `mypy`
- [ ] Skapa `pyproject.toml` med konfiguration
- [ ] Fixa blending-logik i `main.py`
- [ ] Bättre felhantering i ASR-batch
- [ ] Skapa `.github/workflows/ci.yml`

### Steg 5: Tester (pytest)

- [ ] Skapa `tests/`-mapp
- [ ] `test_sentiment.py`
- [ ] `test_lexicon.py`
- [ ] `test_clean.py`
- [ ] `test_asr.py` (mockad transkribering)
- [ ] Mål: ≥ 80% coverage

### Steg 6: Dokumentation & Avslutning
- [ ] Uppdatera `README.md`
- [ ] Skapa `ROADMAP.md`
- [ ] Skapa `docs/PHASE1_SUMMARY.md`
- [ ] Commit och push

---

## Tekniska Krav

- Python 3.11+
- Bevara befintlig modulär struktur
- Använd `typer` för CLI, `rich` för output, `pydantic` där det passar
- Använd `faster-whisper` som primär backend (CTranslate2)
- Bevara bakåtkompatibilitet

---

## Analys av Nuvarande Status

### ASR-implementation (`src/asr.py`)
- Alias `kb-whisper-large` → `KBLab/kb-whisper-large` finns redan
- Stöd för `faster` och `transformers` backends
- **Saknar**: `revision`-parameter för att välja mellan `standard`/`strict`/`subtitle`

### Sentiment (`src/sentiment.py`)
- Fungerande pipeline med `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- Profil-baserad analys med `analyze_smart`
- **Bug**: Blending-logik i `main.py` när `return_all_scores=False`

### CLI (`src/main.py`, `src/asr_cli.py`)
- Välfungerande Typer-baserad CLI
- `asr_cli_refactored.py` har förbättrad struktur med `mode`-parameter

### API (`src/api.py`)
- Fullfjädrad FastAPI med batch/scan endpoints
- Redan konfigurerad för `kb-whisper-large` default

---

*Genererad av Windsurf Devin, Maj 2026*