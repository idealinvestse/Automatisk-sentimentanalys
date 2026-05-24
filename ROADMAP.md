# Projektplan (Roadmap)

## Översikt

Projektet "Automatisk-sentimentanalys" utvecklas i faser med tydliga mål per fas.

---

## Fas 1: Stabilisering & KB-Whisper Integration (Maj–Juni 2026) ✅

**Status**: Genomförd

### Genomförda leveranser
- [x] Utvärderingsramverk (`src/evaluate.py`) med svenskt testset (210 sampel)
- [x] KB-Whisper integration med revision-stöd (`standard`, `strict`, `subtitle`)
- [x] CI/CD pipeline (GitHub Actions: pytest, lint, docker build)
- [x] Kodkvalitet: ruff, black, mypy konfigurerade
- [x] Testtäckning: 76 enhetstester över kärnmoduler
- [x] Bugfix: Lexikon-blending tvingar `return_all_scores=True`
- [x] Uppdaterad dokumentation (README.md, PHASE1_PLAN.md)

---

## Fas 2: Domänanpassning (Planerad: Juli–Augusti 2026)

### Mål
- Samla in och annotera domänspecifik träningsdata (call center-samtal)
- Finetuna sentimentmodell på svensk call center-data
- Utvärdera och publicera resultat

### Planerade uppgifter
- [ ] Samla call center-transkriptioner (anonymiserade)
- [ ] Skapa annoteringsriktlinjer för sentiment i samtalskontext
- [ ] Finetuna `cardiffnlp/twitter-xlm-roberta-base-sentiment` på domändata
- [ ] Jämföra prestanda mot baseline (Fas 1-resultat)
- [ ] Publicera modell på Hugging Face Hub

---

## Fas 3: Produktionisering (Planerad: September–Oktober 2026)

### Mål
- Bygga produktionsredo pipeline med övervakning
- Implementera streaming-API för realtidsanalys
- Optimera för throughput och latency

### Planerade uppgifter
- [ ] Streaming endpoint (WebSocket/SSE)
- [ ] Model versioning och A/B-testning
- [ ] Övervakning: Prometheus-metriker, alerting
- [ ] Lasttestning och prestandaoptimering
- [ ] Produktionsdeployment (Kubernetes/cloud)

---

## Fas 4: Utökade språkstöd (Planerad: November–December 2026)

### Mål
- Stöd för norska och danska
- Gemensam skandinavisk sentimentmodell

### Planerade uppgifter
- [ ] Utvärdera KB-Whisper på norska/danska
- [ ] Samla/skapa testset för norska och danska
- [ ] Finetuna eller adapter-baserad multispråksmodell
- [ ] Uppdatera API för språkval

---

## Teknisk Skuld & Förbättringar

- [ ] Migrera från `setup.py` till full `pyproject.toml`-baserad byggning
- [ ] Lägg till pre-commit hooks
- [ ] Förbättra ASR-felhantering för långa ljudfiler
- [ ] Dokumentera API med OpenAPI-exempel
- [ ] Prestandabenchmark mellan faster-whisper och transformers

---

*Senast uppdaterad: Maj 2026*