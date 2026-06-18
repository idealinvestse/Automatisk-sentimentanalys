# Utvecklingsplan: Backend-förstärkningar och nya funktioner för Call Center Intelligence (Fas 4)

> **Status: SLUTFÖRD** (2026-06-19, v0.4.1)  
> Validering: Fas 1 i `docs/GROK_BUILD_PLAN_FAS1-3.md` (509 tester, 86 %+ coverage).  
> Slutdokumentation: `docs/FAS4_COMPLETION.md`.  
> **Nästa steg:** Data/finetuning (Fas 2 i produktroadmap) + NiceGUI dashboard-visualisering (Fas 3 i GROK-plan) + produktion.

**Version:** 1.1 (validerad & förbättrad)  
**Datum:** 2026-06-02  
**Förutsättning:** Fas 1–3 i tidigare planer är fullt implementerade (inklusive Mistral/OpenRouter LLM-lager för holistisk analys, role inference, structured output, caching och hybrid-arkitektur).  
**Fokus:** Backend-kod för de högst prioriterade funktionerna som rekommenderats efter jämförelse med branschledare (Observe.AI, Cresta, Dialpad, Genesys, NICE, CallMiner m.fl.).  
**Mål:** Lyfta systemet från bra teknisk grund till en komplett, action-oriented call center intelligence-plattform med stark backend-stöd för både nuvarande och framtida dashboard/UI.

**Validering & förbättringar i v1.1 (baserat på 2026 branschpraxis):**
- Validerad mot aktuella källor: Auto QA med 100% coverage + customizable scorecards är table stakes. Agent performance kombinerar kvantitativa metrics + kvalitativa (empathy, de-escalation). Semantic search på transcripts är etablerat. Early PII redaction + loggning är best practice för GDPR.
- Tydligare scorecard-definition med konkret YAML-exempel.
- Mer specifik teknisk approach för Semantic Search (embeddings, hybrid search, self-hosted alternativ).
- Utökad pre-computation & caching-strategi med konkreta exempel.
- Starkare explicita integrationer med befintlig `CallAnalysisReport`, `mistral_analyzer.py` och `pipeline.py`.
- Ny sektion för mätbarhet/KPIs.
- Tydligare prioritering (Fas 4.1 + 4.2 först för högst ROI) och distinktion post-call vs near-real-time.

---

## 1. Bakgrund och nuläge (efter Fas 3)

Med Fas 1–3 implementerade har systemet:
- Stabil svensk ASR + diarization + role inference.
- Per-segment local analysis + holistisk Mistral-baserad analys (trajectory, refined aspects, root cause, actionable summary, agent assessment).
- Hybrid-arkitektur med caching och fallback.
- Strukturerad `CallAnalysisReport` med rik output.
- CLI + REST API + grundläggande Streamlit-dashboard.

**Kvarstående gap mot branschstandard (2026):**
- Agent performance metrics och scorecards är rudimentära.
- Compliance/QA auto-scoring (100% coverage) och structured agent assessment saknas eller är begränsade.
- Avancerad aggregering (hot topics, semantic search, trendanalys) är svag.
- Privacy-hantering (early PII redaction) och alerting/workflows är inte industrialiserade.
- Backend-stöd för drill-down dashboards (pre-computed aggregates, searchable index) är otillräckligt.

**Syfte med Fas 4:**
Implementera skalbar, modulär backend-kod som möjliggör de mest efterfrågade call center-funktionerna med fokus på **actionable, evidence-based output**.

---

## 2. Övergripande strategi (uppdaterad v1.1)

- **Bygg på befintlig struktur:** Använd `CallAnalysisReport`, Pydantic-scheman från Fas 3, `pipeline.py`, `profiles.py` och `mistral_analyzer.py` som bas. Alla nya moduler producerar output som kan mergas in i `CallAnalysisReport`.
- **Modulärt och återanvändbart:** Nya motorer (`agent_performance.py`, `compliance_qa.py`, `insights_aggregator.py`, `semantic_search.py`) kan anropas både inline i pipeline och i batch-jobb.
- **Prestanda & skalbarhet:** Pre-computation av vanliga aggregates + smart caching (Redis eller filbaserad) + indexing för sökning.
- **Privacy by design:** `pii_redactor.py` körs tidigt i pipelinen (före LLM-anrop) och är valbar via profil.
- **Action-oriented + evidensbaserad:** Alla nya funktioner returnerar structured data med evidence_spans och actionable recommendations (inte bara siffror).
- **Hybrid & kostnadseffektivt:** Använd Mistral/OpenRouter selektivt för nyanserade uppgifter (t.ex. detailed coaching recommendations, complex root cause). Regelbaserat + LLM-hybrid för QA scoring.
- **Mätbarhet:** Varje ny modul har definierade exempel-KPIs.

---

## 3. Prioriterad roadmap (Fas 4) – v1.1

**Prioriteringslogik v1.1:** Fas 4.1 (Agent Performance) + Fas 4.2 (QA Engine) först – högst ROI för call center-användare (supervisors & QA). Därefter Fas 4.3 (Insights & Search). Fas 4.4 (Privacy & Alerting) parallellt med 4.1–4.2 eftersom PII påverkar LLM-anrop.

### Fas 4.1: Agent Performance & Assessment Engine (2 veckor)

**Task 4.1.1: Role Inference & Agent/Customer Metrics (Core)**
- **Beskrivning:** Full implementation av role inference + metrics.
- **Specifikation:**
  - Utöka `role_classifier.py` med features: talk_ratio, lexical_formality, question_density, sentiment_variance, intervention_count.
  - Skapa Pydantic-modeller `AgentMetrics` och `CustomerMetrics`.
  - Per samtal: `empathy_score` (0-1), `de_escalation_effectiveness`, `compliance_flags` (lista), `talk_listen_ratio`, `intervention_count`.
  - Agent-nivå aggregering (trender över tid, benchmarking mot team).
- **Påverkade filer:** `src/role_classifier.py`, ny `src/agent_performance.py`, `src/pipeline.py`, `src/llm/schemas.py`.
- **Acceptance criteria:** Varje `CallAnalysisReport` innehåller `agent_assessment` och `customer_metrics`. Agent-trender kan beräknas över dataset.
- **Estimat:** 4–5 dagar.

**Task 4.1.2: Structured Agent Assessment via Mistral (LLM-förstärkt)**
- **Beskrivning:** Nyanserad bedömning via Mistral.
- **Specifikation:**
  - Ny task i `mistral_analyzer.py`: `"agent_assessment_detailed"`.
  - Output (Pydantic): `empathy_score`, `strengths` (lista), `weaknesses` (lista), `specific_coaching_recommendations` (med evidence_spans), `overall_assessment`.
  - Merge med lokala metrics från Task 4.1.1.
- **Acceptance:** Coaching recommendations är specifika, evidensbaserade och actionabla (inte generiska mallar).
- **Estimat:** 3 dagar.

### Fas 4.2: Compliance & QA Auto-Scoring Engine (1,5–2 veckor)

**Task 4.2.1: Compliance & QA Scoring Engine**
- **Beskrivning:** Ny modul `src/compliance_qa.py` med stöd för customizable scorecards.
- **Specifikation:**
  - Scorecards definieras i YAML/JSON under `configs/qa_scorecards/` (exempel nedan).
  - Varje kriterium har: id, description, weight, detection_method (rule-based | llm | hybrid).
  - Auto-scoring: Regelbaserat för explicita kriterier + Mistral för nyanserade (t.ex. "visar empati").
  - Output: `overall_qa_score` (0-100), `passed_criteria`, `failed_criteria`, `compliance_flags`, `risk_level`, `evidence`.
- **Exempel på scorecard (YAML):**
  ```yaml
  name: "Standard_Support_QA_v1"
  version: "1.0"
  criteria:
    - id: "greeting"
      description: "Agent hälsar kunden vänligt"
      weight: 10
      detection_method: "rule-based"
      keywords: ["hej", "god dag", "välkommen"]
    - id: "empathy"
      description: "Agent visar empati vid frustration"
      weight: 20
      detection_method: "llm"
      prompt_hint: "Bedöm om agenten använder fraser som 'jag förstår', 'beklagar' etc."
  ```
- **Påverkade filer:** Ny `src/compliance_qa.py`, `configs/qa_scorecards/`, `src/pipeline.py`, `src/llm/analyzer.py`.
- **Acceptance criteria:** Kan definiera scorecard → auto-score på samtal → få strukturerad output med evidens.
- **Estimat:** 5–6 dagar.

**Task 4.2.2: Pipeline Integration**
- **Beskrivning:** Anropa QA Engine i `CallAnalysisPipeline`.
- **Specifikation:** Lägg till `qa_results` i `CallAnalysisReport`. Stöd per-segment och call-level.
- **Acceptance:** QA-resultat syns i API/CLI och kan trigga alerts.
- **Estimat:** 2 dagar.

### Fas 4.3: Advanced Insights, Search & Aggregation (2 veckor)

**Task 4.3.1: Insights Aggregator & Hot Topics Engine**
- **Beskrivning:** Ny `src/insights_aggregator.py`.
- **Specifikation:**
  - Aggregera: hot_topics (volym + genomsnittlig sentiment + trend), root_cause_clusters, sentiment_trends över tid, top_issues per team/agent.
  - Använd svenska sentence-transformers embeddings + clustering (HDBSCAN).
  - Stöd time-based och agent-based views.
  - Kan anropa Mistral för cluster-beskrivningar.
- **Påverkade filer:** Ny modul, `src/pipeline.py` (batch), `src/llm/analyzer.py`.
- **Acceptance:** Kan generera "Top 10 hot topics förra veckan med volym, sentiment och trend".
- **Estimat:** 5 dagar.

**Task 4.3.2: Semantic Search Engine**
- **Beskrivning:** Ny `src/semantic_search.py`.
- **Specifikation (teknisk approach v1.1):**
  - Embedda transkript + insights (rekommenderad modell: svensk/paraphrase-multilingual sentence-transformers eller KBLab-modell).
  - Lagring: Enkel FAISS (lättvikt, self-hosted) eller Chroma/Qdrant för mer avancerad setup.
  - Hybrid search: Kombinera keyword (BM25) + vector similarity.
  - Stöd filter + natural language queries.
  - Returnera ranked resultat med highlights och relevance score.
- **Acceptance:** Semantic queries ("samtal där kunden var frustrerad över fakturering och agent visade låg empati") ger relevanta träffar.
- **Estimat:** 5–6 dagar.

### Fas 4.4: Privacy, Alerting & Workflows (1,5 veckor)

**Task 4.4.1: PII Redaction Module (Early Pipeline)**
- **Beskrivning:** Ny `src/pii_redactor.py`.
- **Specifikation:**
  - Regelbaserad + NER (svenska modeller) för namn, personnummer, adresser, kortnummer etc.
  - Körs tidigt i pipeline (före LLM-anrop och persistens).
  - Loggar vad som redigerats.
  - Valbar via profil (`anonymize_before_llm: true`).
- **Acceptance:** Känslig data är maskerad i både lokal analys och LLM-anrop. Loggning finns.
- **Estimat:** 4 dagar.

**Task 4.4.2: Alerting & Workflow Engine**
- **Beskrivning:** Ny `src/alerting.py`.
- **Specifikation:**
  - Regelbaserade alerts (t.ex. `customer_sentiment < -0.7 AND escalation_risk > 0.6` → flagga supervisor + skapa coaching task).
  - Stöd webhook-notifieringar och enkla interna workflows.
  - Kan triggas från `insights_aggregator` (trend-baserade alerts).
- **Acceptance:** Kan definiera regel → trigga alert vid matchning.
- **Estimat:** 4–5 dagar.

### Fas 4.5: Prestanda, Caching & API (1 vecka)

**Task 4.5.1: Pre-computation & Advanced Caching**
- **Beskrivning:** Utöka caching från Fas 3.
- **Specifikation:**
  - Pre-computa: Dagliga/veckovisa agent metrics aggregates, hot topics per team, sentiment trends.
  - Lagring: Redis (rekommenderat) eller filbaserad.
  - Smart invalidation vid nya samtal eller schemalagt.
- **Acceptance:** Vanliga dashboard-frågor (t.ex. "agent performance senaste 7 dagarna") är snabba även på stora dataset.
- **Estimat:** 3–4 dagar.

**Task 4.5.2: Utökade REST API-endpoints**
- **Beskrivning:** Nya endpoints i FastAPI.
- **Specifikation:** `/agent_performance/{agent_id}`, `/search/semantic`, `/insights/hot_topics`, `/qa/score`, batch-aggrering.
- **Acceptance:** Alla nya funktioner exponeras via väldokumenterad API.
- **Estimat:** 3 dagar.

---

## 4. Mätbarhet & Exempel-KPIs (ny sektion v1.1)

Varje ny modul bör ha definierade framgångsmått:

- **Agent Performance Engine:** % av samtal med specifika coaching recommendations; minskning i manuell QA-tid; korrelation mellan empathy_score och CSAT.
- **QA Scoring Engine:** % auto-scored interactions (mål: 100%); consistency (inter-rater agreement med mänsklig QA); compliance risk reduction.
- **Insights Aggregator:** Tid till insikt (t.ex. hur snabbt "hot topic" identifieras); användning av hot topics i processförbättringar.
- **Semantic Search:** Precision@10 på test queries; tid till relevant samtal.
- **PII Redaction:** % korrekt redigerad PII (precision/recall); loggade incidenter.

---

## 5. Nya moduler / filer (v1.1)

- `src/agent_performance.py`
- `src/compliance_qa.py`
- `src/insights_aggregator.py`
- `src/semantic_search.py`
- `src/pii_redactor.py`
- `src/alerting.py`
- `configs/qa_scorecards/` (med exempel)
- Utökningar i `src/llm/schemas.py`, `src/pipeline.py`

---

## 6. Risker & Mitigation (uppdaterad)

- **Prestanda på stora volymer:** Pre-computation + indexing från början.
- **Kvalitet på auto-QA och coaching:** Börja hybrid (regel + Mistral) och iterera på prompts/scorecards.
- **Komplexitet:** Håll modulerna löst kopplade med tydliga interfaces.
- **Privacy:** PII-redaction tidigt + loggning + valbarhet.

---

## 7. Hur man använder planen (v1.1)

1. Placera filen i projektroten.
2. Använd med dedikerad Grok Build-prompt (se separat fil).
3. Börja med **Fas 4.1 + 4.2** (högst ROI).
4. Efter varje task: uppdatera status-tabellen och kör tester + evaluate.

**Statusöversikt (v1.1)**

| Fas | Task | Status | Start | Klart | Notes |
|-----|------|--------|-------|-------|-------|
| 4.1 | 4.1.1 Role Inference & Metrics | DONE | 2026-06-03 | 2026-06-03 | Implemented: extended role_classifier.py (features), new src/agent_performance.py (Pydantic AgentMetrics/CustomerMetrics/CallAgentPerformance + compute + aggregate), explicit calls + merge in pipeline.py (both audio/segments paths), updated schemas.py (enhanced AgentAssessment + new metric models), compat in mistral. Local metrics + customer_metrics + agent_assessment always in report.results. Hybrid merge point documented. |
| 4.1 | 4.1.2 LLM Agent Assessment | DONE | 2026-06-03 | 2026-06-03 | Implemented: added 'agent_assessment_detailed' to SUPPORTED + defaults, forward agent_performance_local in local_ctx to LLM (for hybrid merge), strengthened prompts.py with evidence-based specific_coaching + use of local metrics, schema already extended in 4.1.1 supports weaknesses/specific recs w/ evidence_spans/overall. Pipeline merge logic (LLM over local) covers. Tests updated. Coaching recs now guided to be specific/actionable. |
| 4.2 | 4.2.1 QA Scoring Engine | DONE | 2026-06-03 | 2026-06-03 | Implemented: full src/compliance_qa.py with QAScorer, load_scorecard, hybrid (rule for greeting/closing, hybrid/llm for empathy/resolution), Pydantic models with evidence_spans, default scorecard yaml. Explicit integration+merge+LLM-log in pipeline.py for both paths (4.2.2). Tests + standalone verified. Actionable output (flags, risk, per-crit evidence). |
| 4.2 | 4.2.2 Pipeline Integration | DONE | 2026-06-03 | 2026-06-03 | Done together with 4.2.1: qa and compliance_qa in results for every report; local_signals include agent metrics; LLM criteria logged. |
| 4.3 | 4.3.1 Insights Aggregator | DONE | 2026-06-03 | 2026-06-03 | Implemented: new src/insights_aggregator.py (Pydantic HotTopic/AggregatedInsights using EvidenceSpan, optional embeddings+HDBSCAN with keyword fallback, Mistral for descriptions when analyzer provided - documented, aggregates hot_topics w/ volume/sentiment/trend/evidence, root clusters, agent issues). Explicit aggregate_insights() in pipeline.py (batch, shows call site + merge). Added to evaluate.py. Tests + smoke verified. Caching/privacy/hybrid per rules. |
| 4.3 | 4.3.2 Semantic Search | DONE | 2026-06-03 | 2026-06-03 | Implemented: new src/semantic_search.py (hybrid keyword+vector, optional FAISS, builds index from reports, SearchHit with highlights/evidence, graceful no-dep fallback). Explicit semantic_search() on pipeline with example. Tests pass. Consistent with aggregator embeddings. |
| 4.4 | 4.4.1 PII Redaction | DONE | 2026-06-03 | 2026-06-03 | Implemented: enhanced src/llm/pii_redactor.py (detailed events logging + PiiRedactionLog Pydantic, more patterns (cc, address, name heuristic), optional NER stub, return_log support, idempotent). Explicit early integration in pipeline.py (both audio+segments paths, BEFORE run_analyzers/LLM/persist, redacted text for local+LLM, log in results['pii_redaction']). Profile-driven via existing key. evaluate.py extended. Tests + end-to-end verification (enabled path produces log + redacted segments). Matches acceptance + privacy-by-design. |
| 4.4 | 4.4.2 Alerting & Workflows | DONE | 2026-06-03 | 2026-06-03 | Implemented: new src/alerting.py (DEFAULT_RULES + engine, evidence_spans, recommended_actions like flag_supervisor/create_coaching, check + check_from_aggregate + webhook stub, run_alerts_on_results helper). Explicit calls + merge in pipeline.py (per call after qa/pii, and in aggregate_insights for trends). Pydantic Alert/AlertSummary in schemas. evaluate extension + tests (6 passed). Matches "definiera regel → trigga alert". |
| 4.5 | 4.5.1 Pre-computation & Advanced Caching | DONE | 2026-06-03 | 2026-06-03 | Implemented: new src/caching.py (AggregateCache with file + optional Redis, precompute_and_cache, _is_valid TTL, invalidate). precompute_agent_aggregates + precompute_hot_topics using existing logic. Explicit get_cached_* + invalidate in pipeline.py (with invalidation strategy docs). Enhanced aggregate_insights to use cache. Tests + evaluate. (Note: 4.5.2 API endpoints recommended next for full exposure.) |
| 4.5 | 4.5.2 Utökade REST API-endpoints | DONE | 2026-06-03 | 2026-06-03 | Implemented: added 5 new endpoints in src/api/routers/pipeline.py (/agent_performance/{agent_id}, /search/semantic, /insights/hot_topics, /qa/score, /alerts) with request/response Pydantic in schemas.py. Explicit use of pipe.get_cached_*, semantic_search, aggregate etc. + results merge. Updated tests/test_api.py (5 passed). All Fas4 features now API-exposed per plan. |

---

*Denna v1.1-plan är validerad mot aktuell branschpraxis 2026 och är redo för implementation. Den ger stark backend-stöd för en modern, action-oriented call center intelligence-plattform.*

---

**Slut på plan v1.1.**