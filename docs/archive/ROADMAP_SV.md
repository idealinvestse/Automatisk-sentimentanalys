"""
# Projektplan (Roadmap) — historisk svensk översikt

> **Aktuell status och stubs:** se [docs/ROADMAP.md](docs/ROADMAP.md) (canonical).  
> **Städplan:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md)

## Översikt

Projektet "Automatisk-sentimentanalys" utvecklas i faser med tydliga mål per fas.

---

## Fas 6.0: Dashboard-migrering till NiceGUI ✅ Klar

**Mål**: Migrera den befintliga Streamlit-dashboarden till NiceGUI för bättre reaktivitet, modern UX och långsiktig skalbarhet.

**Officiell plan**: Se `docs/archive/MIGRATION_TO_NICEGUI_PLAN.md` (detaljerad guide + task breakdown, 5 faser, 8–12 veckor).

**Nyckel-fördelar**:
- Event-driven reaktivitet (bättre än Streamlit reruns)
- Starkare state management
- Naturlig integration med befintlig FastAPI-backend
- Modern Tailwind-baserad UI
- Bättre stöd för real-time (transkriberings-progress, logs)

**Status**: ✅ Migrering klar (Fas 5) — NiceGUI är standarddashboard (`app/nicegui_dashboard/`). Streamlit call center-dashboard borttagen. `DASHBOARD_UI=nicegui` via `app/dashboard_launcher.py`.

**Drift**: `sentimentanalys-dashboard` eller `docker compose -f docker-compose.nicegui.yml up`

**Kvar att städa**: `app/setup_hub.py` (Streamlit), launcher-dropdown — se `docs/CLEANUP_PLAN.md` Fas 2.3.

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

## Fas 1–3: Avancerad Call Center Intelligence (Slutfört 2026)

Detaljerad plan och implementation finns i `docs/archive/UTVECKLINGSPLAN.md` (root).

### Genomförda leveranser (utöver Fas 1–2 grund)
- **Task 1.1–1.4 (ASR)**: WhisperX-backend, hotwords + initial_prompt (auto från configs/callcenter_hotwords.txt), chunking+low_confidence+lexicon-boost, valbar preprocess (--preprocess).
- **Task 1.5 (ABSA)**: src/analysis/aspect.py (hybrid), registrerad, callcenter-aspects i profile.
- **Fas 2 (2.1–2.5)**: emotion, role inference, trajectory/escalation, spoken_normalizer, llm_judge-stub. Alla via analysis/ registry.
- **Fas 3 (LLM)**: Full Mistral/OpenRouter hybrid (europeisk-first): OpenRouterClient med strict schema + cache + GDPR-logs, ConversationMistralAnalyzer + schemas/prompts, hybrid merge i pipeline, CLI/API/dashboard/evaluate-stöd, dedikerade docs (FAS3_MISTRAL_LLM_INTEGRATION.md). PII-redactor stub + human study template som follow-up.
- **Review + commit**: Strukturerad self-review (docs/archive/REVIEW_MISTRAL_FAS3.md) + commit med detaljerat meddelande. Alla plan-regler följda.

Se även uppdaterad `docs/ARCHITECTURE.md`, `docs/FAS3_MISTRAL_LLM_INTEGRATION.md` och README.

### Nästa steg (produktion + iteration)
- Samla riktiga anonymiserade calls + finetuna (LoRA)
- Bygg ut dashboard med aspects, emotions, trajectory plots + LLM-sektioner
- Streaming / realtids-API (WebSocket)
- Full utvärdering + human correlation på callcenter-data (använd `llm-human-study` mallen)
- PII-redaction i produktion (utöka stuben)
- v0.4 release efter Fas 3 (inkl. Mistral)

---

## Teknisk Skuld & Förbättringar

- [x] Migrera från `setup.py` till full `pyproject.toml`-baserad byggning
- [ ] Lägg till pre-commit hooks
- [ ] Förbättra ASR-felhantering för långa ljudfiler
- [ ] Dokumentera API med OpenAPI-exempel
- [ ] Prestandabenchmark mellan faster-whisper och transformers

---

*Senast uppdaterad: Juni 2026 – Se `docs/archive/MIGRATION_TO_NICEGUI_PLAN.md` för den nya Fas 6.0 dashboard-migreringen.*
"""