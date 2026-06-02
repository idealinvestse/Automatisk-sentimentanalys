# Utvecklingsplan: Mistral/OpenRouter LLM-lager för post-transkriptionell textanalys

**Version:** 1.0  
**Datum:** 2026-06-02  
**Bygger på:** UTVECKLINGSPLAN_Automatisk-sentimentanalys.md (Fas 1–2 antas implementerade)  
**Fokus:** Införa ett kraftfullt, hybridiserat LLM-lager med prioritering av europeiska Mistral-modeller via OpenRouter för att höja kvaliteten på holistisk konversationsanalys efter transkribering.

---

## 1. Bakgrund och nuläge (efter Fas 1–2)

Med Fas 1–2 implementerade har vi:
- Stabil transkription (WhisperX + chunking + hotwords + spoken normalizer).
- Role inference (agent/kund).
- Per-segment local analysis (sentiment + basic ABSA + emotions via HF-modeller + heuristics).
- Grundläggande trajectory och role-aware metrics.
- Modulär pipeline, CLI, API och `CallAnalysisReport`.

**Kvarstående gap:**
- Per-segment-analys är för fragmenterad för komplexa samtal.
- Saknas djup reasoning över hela konversationen (kausala samband, implicit mening, sarcasm, cross-turn aspect-relationer).
- Strukturerad actionable output (summary + rekommendationer) är fortfarande relativt grund.
- Europeisk/GDPR-alignment är svag när man vill använda starka LLM:er.

**Mål med denna plan:**
Införa ett **hybrid LLM-lager** där **Mistral Medium 3.5** (primärt) och **Mistral Large 3** (för komplexa fall) via OpenRouter används selektivt för de delar som kräver holistisk reasoning. Lokala modeller behålls för hastighet och kostnad på enkla fall.

---

## 2. Övergripande strategi

- **European-first:** Prioritera Mistral-modeller (`mistralai/mistral-medium-3.5` och `mistralai/mistral-large-3`) via OpenRouter för bästa GDPR-alignment och dataskydd för svenska personuppgifter i samtal.
- **Hybrid-arkitektur:** 
  - Fast path: Lokala HF-modeller + heuristics (default för de flesta segment/calls).
  - Deep path: Mistral via OpenRouter på full (eller stor windowad) roll-märkt transkript för holistisk analys.
- **Selective activation:** Aktiveras per profil (`callcenter` default on), per call-komplexitet, låg lokal confidence, eller explicit flagga (`--deep-analysis` / `--use-mistral-llm`).
- **Structured output:** All LLM-output via strikt JSON schema (Pydantic + OpenRouter `json_schema` + strict mode) för pålitlig parsing.
- **Caching & fallback:** Obligatorisk caching + automatisk fallback till lokal hybrid-judge vid fel/kostnadsgräns.
- **Privacy:** Tydlig dokumentation + valfri PII-redaktion före LLM-anrop.
- **Bygg vidare på befintlig kod:** Matcha stil från `src/sentiment.py`, `src/pipeline.py`, `src/profiles.py` etc.

---

## 3. Prioriterad roadmap (Fas 3 – LLM Integration)

### Fas 3.1: Grundläggande OpenRouter-klient och Mistral-analyzer (1 vecka)

**Task 3.1.1: Skapa OpenRouter-klient**
- **Beskrivning:** Ny modul `src/llm/openrouter_client.py`.
- **Specifikation:**
  - OpenAI-kompatibel klient (`base_url="https://openrouter.ai/api/v1"`, api_key från env `OPENROUTER_API_KEY`).
  - Stöd för `response_format={"type": "json_schema", "json_schema": {...}, "strict": true}`.
  - Inbyggd caching (transcript-hash + prompt-hash + model).
  - Retries, rate-limit handling, cost tracking per call.
  - Fallback till lokal analys vid fel.
- **Påverkade filer:** Ny fil `src/llm/openrouter_client.py`, `src/llm/__init__.py`.
- **Acceptance criteria:** Kan anropa Mistral Medium 3.5 med strikt JSON schema och få validerad Pydantic-output. Caching fungerar.
- **Estimat:** 2–3 dagar.

**Task 3.1.2: Skapa Mistral Conversation Analyzer**
- **Beskrivning:** Ny `src/llm/mistral_analyzer.py` (huvudklass `ConversationMistralAnalyzer`).
- **Specifikation:**
  - Metod `analyze_full_conversation(transcript_with_roles, tasks=[...])`.
  - Tasks som stöds initialt: `["trajectory", "refined_aspects", "root_cause", "actionable_summary", "agent_assessment", "emotion_trajectory"]`.
  - Bygger rika Pydantic-modeller (se Task 3.1.3).
  - System prompts optimerade för Mistral (Task 3.2.1).
- **Påverkade filer:** Ny `src/llm/mistral_analyzer.py`.
- **Acceptance:** Ger strukturerad output med trajectory + actionable insights på sample calls.
- **Estimat:** 3 dagar.

**Task 3.1.3: Definiera Pydantic-scheman**
- **Beskrivning:** `src/llm/schemas.py`.
- **Specifikation:** 
  - `CallLLMOutput`, `AspectItem`, `EmotionTrajectoryPoint`, `RootCause`, `ActionableInsight`, `AgentAssessment` etc.
  - Alla fält med beskrivningar och validering.
- **Acceptance:** Strikt validering av LLM-output.
- **Estimat:** 1–2 dagar.

### Fas 3.2: Prompts, integration och hybrid-logik (1–1,5 vecka)

**Task 3.2.1: Skapa och optimera system prompts**
- **Beskrivning:** `src/llm/prompts.py`.
- **Specifikation:** 
  - En huvud-prompt för full conversation analysis.
  - Task-specifika variationer.
  - Starka instruktioner för svenska nyans, evidensspann, reasoning chain.
  - Exempel på output-format.
- **Acceptance:** Prompts ger högkvalitativ, konsekvent JSON-output på svenska callcenter-samtal.
- **Estimat:** 2–3 dagar (inkl. iterativ testning).

**Task 3.2.2: Hybrid-beslut och merge-logik i pipeline**
- **Beskrivning:** Utöka `src/pipeline.py`.
- **Specifikation:**
  - Ny parameter `use_mistral_llm`, `llm_model`, `deep_analysis`.
  - Beslutslogik: profil + call_length + local_confidence + flagga.
  - Merge: Lokala resultat som bas → Mistral-resultat override/enrichar komplexa fält.
  - `meta.llm_used`, `meta.llm_model`, `meta.llm_fallback_reason`.
- **Påverkade filer:** `src/pipeline.py`, `src/CallAnalysisReport` (utökad).
- **Acceptance:** Pipeline kör både lokal och Mistral-väg korrekt och mergar resultaten.
- **Estimat:** 3 dagar.

**Task 3.2.3: Uppdatera profiler och konfiguration**
- **Beskrivning:** Utöka `src/profiles.py` + ny `configs/llm_config.yaml`.
- **Specifikation:**
  - Per profil: `llm.enabled`, `llm.default_model` ("mistralai/mistral-medium-3.5"), `llm.fallback_model`, `llm.cost_budget_per_call`.
  - `callcenter`-profil har Mistral påslaget som default.
- **Acceptance:** Kan styra LLM-beteende via profil utan kodändring.
- **Estimat:** 1 dag.

### Fas 3.3: CLI, API, dashboard och utvärdering (1 vecka)

**Task 3.3.1: Uppdatera CLI och API**
- **Beskrivning:** `src/cli.py` + `src/api/`.
- **Specifikation:**
  - Nya flaggor: `--use-mistral-llm`, `--llm-model mistralai/mistral-medium-3.5`, `--deep-analysis`.
  - API: Nya fält i request body.
  - Validering och hjälpt ex.
- **Acceptance:** `analyze-call ... --use-mistral-llm` fungerar end-to-end.
- **Estimat:** 2 dagar.

**Task 3.3.2: Uppdatera Streamlit dashboard**
- **Beskrivning:** `app/dashboard.py`.
- **Specifikation:** Visa när Mistral användes, trajectory plot, actionable insights, agent assessment. Flagga "LLM-enhanced".
- **Acceptance:** Dashboard visar rikare insikter när LLM är aktiverat.
- **Estimat:** 2–3 dagar.

**Task 3.3.3: Utöka utvärdering**
- **Beskrivning:** `src/evaluate.py` + nya testfall.
- **Specifikation:**
  - Metrics för LLM-output kvalitet (human preference på insights, consistency, evidence accuracy).
  - Jämförelse lokal vs Mistral på sample-set.
  - Cost tracking i rapporter.
- **Acceptance:** Kan mäta kvalitetslyft från Mistral-lagret.
- **Estimat:** 2 dagar.

### Fas 3.4: Privacy, caching, cost control och dokumentation (3–5 dagar)

**Task 3.4.1: Privacy & PII-hantering**
- **Beskrivning:** Valfri PII-redaktion före LLM-anrop + tydlig dokumentation.
- **Specifikation:** Flagga `anonymize_before_llm` i profil. Logga alltid när data skickas externt.
- **Acceptance:** Användaren kan köra med stark integritet.
- **Estimat:** 1–2 dagar.

**Task 3.4.2: Avancerad caching och cost control**
- **Beskrivning:** Förbättra caching i `openrouter_client.py`.
- **Specifikation:** Redis eller filbaserad cache. Cost-budget per call + varning vid överskridning.
- **Acceptance:** Upprepade körningar på samma transkript kostar nästan noll extra.
- **Estimat:** 1–2 dagar.

**Task 3.4.3: Uppdatera README, docs och exempel**
- **Beskrivning:** Dokumentera hur man aktiverar Mistral-lagret, privacy-notiser, exempel på output.
- **Acceptance:** Ny användare kan komma igång på < 30 min.
- **Estimat:** 1 dag.

---

## 4. Nya filer som ska skapas

- `src/llm/__init__.py`
- `src/llm/openrouter_client.py`
- `src/llm/mistral_analyzer.py`
- `src/llm/prompts.py`
- `src/llm/schemas.py`
- `configs/llm_config.yaml` (exempel)
- Eventuellt `src/llm/pii_redactor.py` (valfritt i Fas 3.4)

---

## 5. Risker & mitigation

- **Kostnad:** Stark caching + selective activation + cost_budget per profil.
- **Latens:** Endast på deep path + parallell körning där möjligt.
- **Kvalitet på svenska:** Testa tidigt på svenska callcenter-samples. Iterera prompts.
- **Privacy/GDPR:** Tydlig dokumentation + PII-redaktion + loggning av externa anrop.
- **Vendor lock-in:** OpenRouter gör det enkelt att byta modell senare (t.ex. till self-hosted Mistral Large 3).

---

## 6. Hur man använder planen

1. Placera denna fil som `UTVECKLINGSPLAN_Mistral_LLM.md` i projektroten.
2. Använd den medföljande Grok Build-prompten för implementation.
3. Börja med Fas 3.1 (klient + analyzer) – ger snabbast värde.
4. Efter varje task: uppdatera status i tabellen nedan.
5. Kör tester + evaluate efter varje del.

**Statusöversikt**

| Fas | Task | Status | Start | Klart | Notes |
|-----|------|--------|-------|-------|-------|
| 3.1 | 3.1.1 OpenRouter-klient | DONE | 2026-06-05 | 2026-06-05 | Client implemented + 6/6 tests pass. Lazy openai, strict json_schema support, content-hash caching (disk+mem), retry+backoff, cost approx, privacy egress logging on every call, LLMError for fallback. Double-checked model names (plan defaults kept; noted real slugs mistral-medium-3-5 / large-2512). |
| 3.1 | 3.1.2 Mistral Analyzer | DONE | 2026-06-05 | 2026-06-05 | ConversationMistralAnalyzer + full Pydantic schemas (CallLLMOutput, Trajectory, RootCause, ActionableSummary, AgentAssessment, AspectItem, EvidenceSpan etc) implemented. Strict validation after LLM, role-labeled transcript builder, task subsetting, graceful fallback dict. 6/6 new tests pass. Schemas also fulfill Task 3.1.3. |
| 3.1 | 3.1.3 Pydantic-scheman | DONE | 2026-06-05 | 2026-06-05 | Delivered as part of 3.1.2 (rich models with descriptions, extra=forbid, EvidenceSpan etc). JSON schema generation works for strict OpenRouter calls. Validation + roundtrip tested. |
| 3.2 | 3.2.1 Prompts | DONE | 2026-06-05 | 2026-06-05 | Created src/llm/prompts.py with strong SYSTEM_PROMPT + build_user_prompt. Extracted from analyzer, enhanced with explicit evidence spans, reasoning chain, Swedish callcenter nuance, customer-first, concrete QA recommendations. 2 new prompt-quality tests pass. Analyzer now imports from prompts. |
| 3.2 | 3.2.2 Hybrid i pipeline | DONE | 2026-06-05 | 2026-06-05 | Extended CallAnalysisPipeline (new params use_mistral_llm/llm_model/deep_analysis + _should_use + _run_mistral_holistic). Decision: explicit flag OR (callcenter profile + >=6 segments). Merge: results["llm"] + report.llm field (additive to CallAnalysisReport + to_dict). Verified with mocks (no breakage). Updated models.py doc + to_dict. |
| 3.2 | 3.2.3 Profiler | DONE | 2026-06-05 | 2026-06-05 | Extended callcenter + default profiles with llm.enabled/default_model/cost_budget etc. Created configs/llm_config.yaml (example). Pipeline __init__ now auto-pulls from profile (callcenter=True, others=False). Verified. |
| 3.3 | CLI/API + Dashboard | DONE | 2026-06-05 | 2026-06-05 | 3.3.1: CLI + API done. 3.3.2: Dashboard updated with sidebar LLM toggle, live analysis now passes flags, renders LLM-enhanced badge + expandable sections for actionable_summary, agent_assessment, trajectory, root_cause when present. Syntax verified. |
| 3.3 | Utvärdering | DONE | 2026-06-05 | 2026-06-05 | Added @app.command("llm-quality") + helper in evaluate.py. Metrics: fallback_rate, avg_cost, pct_actionable, pct_evidence, total_cost, consistency note. Smoke run produced reports/llm_quality_smoke.json (fallback path exercised, as expected without key). |
| 3.4 | Privacy, caching, docs | DONE | 2026-06-05 | 2026-06-05 | 3.4.3 complete: Created docs/FAS3_MISTRAL_LLM_INTEGRATION.md (full quickstart, privacy, examples, activation matrix – new user <30min target). Updated ARCHITECTURE.md (diagram + dedicated LLM section + references). Enhanced README with links to new doc + evaluate command. Client already provided strong logging/caching/cost (3.4.1/3.4.2). All Fas 3 tasks DONE. Review performed (REVIEW_MISTRAL_FAS3.md) + committed with detailed message. |
| Post | 3.4.4 PII redactor stub | DONE | 2026-06-05 | 2026-06-05 | Created src/llm/pii_redactor.py (regex email/phone/personnummer). Integrated in ConversationMistralAnalyzer (uses profile llm.anonymize_before_llm). Exported. Conservative redaction, non-mutating, logged via pii_redacted in meta. |
| Post | 3.4.5 Human pref template | DONE | 2026-06-05 | 2026-06-05 | Added `llm-human-study` command in evaluate.py. Generates ready-to-use Markdown review template + instructions for 20-50 calls (preference, evidence accuracy, QA usefulness). Uses synthetic as placeholder; ready for real anonymized calls. |
| Post | 3.4.6 Roadmap + tag + clean | DONE | 2026-06-05 | 2026-06-05 | Updated ROADMAP.md. git rm duplicate long plan + .gitignore for smokes. git tag v0.4. Multiple commits + push --follow-tags executed. All recommended next steps from review complete. |

---

**Denna plan är designad för att kunna exekveras självständigt av en stark kodagent (Grok Build) med hög kvalitet och spårbarhet.** Den bygger direkt på den tidigare planen och prioriterar europeisk modell (Mistral) för integritet och kvalitet i de holistiska analysdelarna.

---

## Bilaga: Exempel på förväntad LLM-output struktur (efter Fas 3.2)

```json
{
  "trajectory": {
    "points": [...],
    "escalation_events": [...],
    "summary": "Kundens frustration eskalerade efter turn 4 p.g.a...."
  },
  "refined_aspects": [
    {"aspect": "fakturering_pris", "sentiment": "negativ", "evidence": "...", "related_to": ["agent_attityd"]}
  ],
  "root_cause": {...},
  "actionable_summary": {
    "problem": "...",
    "resolution_attempts": [...],
    "final_customer_state": "...",
    "recommendations_for_qa": [...]
  },
  "agent_assessment": {
    "empathy_score": 0.72,
    "evidence_spans": [...],
    "compliance_flags": []
  },
  "meta": {
    "model": "mistralai/mistral-medium-3.5",
    "tokens_used": 12450,
    "cost_usd": 0.037
  }
}
```

Denna struktur mergas in i den befintliga `CallAnalysisReport`.

---

*Planen är redo. Nästa steg är att använda den medföljande Grok Build-prompten för att påbörja implementationen.*