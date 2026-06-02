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

## Fas 2: Domänanpassning (Pågående/grund levererad)

Se även `docs/PHASE2_PLAN.md` och `docs/FAS2_SUMMARY.md`.

### Mål
- Samla in och annotera domänspecifik träningsdata (call center-samtal)
- Finetuna sentimentmodell på svensk call center-data
- Utvärdera och publicera resultat

### Planerade uppgifter
- [x] Skapa syntetiska call center-exempel (anonymiserade seed-data)
- [x] Skapa dataformat och första annoterings-/labelstruktur
- [x] Skapa LoRA/PEFT fine-tuning pipeline för domändata
- [x] Skapa baseline- och Fas 2-jämförelserapporter

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

---


---

## Teknisk Skuld & Förbättringar

- [x] Migrera från `setup.py` till full `pyproject.toml`-baserad byggning
- [ ] Lägg till pre-commit hooks
- [ ] Förbättra ASR-felhantering för långa ljudfiler
- [ ] Dokumentera API med OpenAPI-exempel
- [ ] Prestandabenchmark mellan faster-whisper och transformers

---

*Senast uppdaterad: Maj 2026*