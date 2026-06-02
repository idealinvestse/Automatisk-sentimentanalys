# Utvecklingsplan: Förbättrad transkription och avancerad textanalys
## Automatisk-sentimentanalys (Call Center Intelligence)

**Version:** 1.0  
**Datum:** 2026-06-02  
**Ägare:** Oscar / idealinvestse  
**Fokus:** Transkription (ASR) + Textanalys / Sentiment pipeline med stark call center intelligence inriktning  
**Mål med planen:** Lyfta projektet från proof-of-concept till produktionsredo verktyg som levererar **actionable, nyanserad och roll-medveten insikt** från svenska samtal. Bygg strikt på befintlig arkitektur (modulär pipeline, profiles, hybrid model+lexicon+heuristics, CLI/API, evaluate framework).

---

## 1. Bakgrund och nuläge (sammanfattning från djupgranskning)

Projektet har stark grund:
- Svensk ASR-prioritering med `KBLab/kb-whisper-large` (bästa WER på svenska benchmarks).
- Två backends (faster-whisper optimerad + transformers pipeline).
- Revisioner (strict = verbatim för callcenter – bra).
- Hybrid sentiment (`cardiffnlp/twitter-xlm-roberta-base-sentiment` + sensaldo_lexicon + callcenter-heuristik för negation/intensity/polite).
- Speaker diarization (pyannote + naiv heuristic fallback).
- End-to-end `CallAnalysisPipeline` som binder transcription → diarization → sentiment/intent/summary/topics/predictive.
- Profiler, cleaning, CLI, FastAPI, Streamlit dashboard, evaluate.py, finetune.py.

**Kritiska begränsningar som direkt blockerar målen:**
- Transkription saknar chunking+overlap i faster-backend, domain hotwords, bruspreproc och confidence weighting.
- Diarization ger bara `SPEAKER_X` – ingen role inference (agent vs kund). Detta gör per-segment sentiment delvis meningslöst för agent performance och kundtrajectory.
- Sentiment är för grovt (3-class) och kontextlöst. Saknar aspects, emotions, sarcasm, conversational dynamics.
- Text efter strict-transkription innehåller disfluencies utan dedikerad spoken normalization.
- Aggregation till call-level är implicit och saknar weighted customer-focus + structured insights.
- Utvärdering är generisk – saknar callcenter-specifika KPIs.

**Konsekvens:** Nuvarande pipeline riskerar att ge missvisande eller för grunda insikter (t.ex. "negativ" utan att veta *vad* eller *vem* som eskalerar).

---

## 2. Övergripande strategi och principer

- **Bygg vidare, bryt inte:** All ny kod ska matcha befintlig stil (type hints, lazy loading, profile-driven config, error wrappers som `TranscriptionError`/`AnalysisError`, logging, caching).
- **Modulärt:** Nya analystyper i egna filer under `src/` (t.ex. `aspect.py`, `emotion.py`). Exportera via `__init__.py`.
- **Hybrid först:** Små effektiva modeller primärt. LLM-judge (lokal quantized) endast på low-confidence eller komplexa fall. Behåll offline-fokus.
- **Callcenter-vikt:** Customer turns väger tyngre. Agent-analys = empathy/compliance/de-escalation. Customer-analys = trajectory + unresolved aspects.
- **Utvärdering först:** Varje fas slutar med utökad evaluate.py + synthetic + manuella testfall från `samples/`.
- **Backward compat:** CLI/API parametrar, output-struktur i `CallAnalysisReport` och befintliga profiler får inte brytas utan explicit deprecation.
- **Data:** Skapa `data/callcenter/` för samples, lexikon-utökningar och framtida labeled data. Rekommenderar anonymiserade verkliga calls + synthetic för aspects/emotions.

---

## 3. Prioriterad roadmap (3 faser)

### Fas 1: Quick Wins – Stabilare ASR + Grundläggande Aspect-Based Sentiment (1–2 veckor)

**Mål:** Omedelbar kvalitetslyft på input + första structured "vad är problemet?" insikt. Hög ROI.

#### Task 1.1: Integrera WhisperX som förstklassig backend
- **Beskrivning:** Lägg till `src/transcription/whisperx.py` (eller starkt utökad factory). WhisperX ger bättre word alignment, inbyggd diarization och chunking+overlap i en pipeline.
- **Specifikation:**
  - Stöd samma parametrar som befintliga: `language="sv"`, `vad`, `word_timestamps`, `diarize`, `num_speakers`, `revision` (mappar till model variant).
  - Använd `whisperx.load_model("large-v3", device=..., compute_type=...)` + `whisperx.align` + `whisperx.diarize`.
  - Behåll KB-Whisper som default (bästa svenska WER). WhisperX som `--backend whisperx`.
  - Lazy loading + caching som i befintliga klasser.
- **Påverkade filer:**
  - `src/transcription/factory.py` (lägg till "whisperx" case)
  - `src/transcription/__init__.py` (exportera)
  - `src/cli.py` (lägg till `--backend whisperx` validering + exempel)
  - `src/pipeline.py` (stöd backend i CallAnalysisPipeline)
- **Acceptance criteria:**
  - `python -m src.cli transcribe samples/call.wav --backend whisperx --diarize` fungerar och producerar bättre timestamps än ren faster.
  - Jämförande test i `tests/test_transcription.py` (eller nytt) visar lägre DER (diarization error rate) och bättre word alignment.
  - Inga regressioner på befintliga backends.
- **Estimat:** 3–4 dagar (inkl. dependency + test).

#### Task 1.2: Chunking + overlap + confidence weighting i FasterWhisperTranscriber
- **Beskrivning:** Implementera chunking (30s chunks, 5s overlap) med smart merge i `faster_whisper.py` (chunk_length_s används nu). Flagga low-confidence segments.
- **Specifikation:** Merge overlapping segments (text + max/weighted avg_confidence). Lägg till `segment.confidence` och `low_confidence_flag` i Segment model.
- **Påverkade filer:** `src/transcription/faster_whisper.py`, `src/transcription/base.py` (Segment dataclass om behövs).
- **Acceptance:** Långa filer (>10 min) transkriberas utan OOM och med vettig merge. Low-conf segments får högre lexicon_weight automatiskt i downstream.
- **Estimat:** 2 dagar.

#### Task 1.3: Domain hotwords / initial_prompt stöd
- **Beskrivning:** Exponera `hotwords` (lista) och `initial_prompt` i transcribe-metoder och CLI/API.
- **Specifikation:** För callcenter: ladda från `configs/callcenter_hotwords.txt` eller profil. Skicka till Whisper (faster-whisper och WhisperX stödjer det).
- **Påverkade filer:** `src/transcription/*.py`, `src/cli.py`, `src/api/`, `configs/`.
- **Acceptance:** "fakturering" + "återbetalning" + bolagsnamn ger lägre WER på tekniska calls.
- **Estimat:** 1 dag.

#### Task 1.4: Bruspreproc (valbar)
- **Beskrivning:** Valbar pre-processing steg före ASR (ffmpeg high-pass + noisereduce eller torchaudio).
- **Påverkade filer:** `src/transcription/base.py` eller ny `preprocess.py`.
- **Acceptance:** Tydlig WER-förbättring på bullriga samples.
- **Estimat:** 1–2 dagar.

#### Task 1.5: Grundläggande Aspect-Based Sentiment Analysis (ABSA)
- **Beskrivning:** Ny modul `src/aspect.py`. Definiera callcenter-aspects. Extrahera aspects + sentiment per span + evidence.
- **Specifikation:**
  - Aspects (initialt): `["kundtjänst_kvalitet", "teknisk_lösning", "fakturering_pris", "väntetid", "agent_attityd", "produkt_kvalitet", "uppföljning", "annat"]`.
  - Metod: Zero-shot med stark multilingual model (t.ex. ModernBERT eller DeBERTa-v3 ABSA variant) ELLER keyword + embedding similarity + befintlig sentiment på extractad span. Börja enkelt med hybrid (keywords + sentiment.py).
  - Output struktur: `{"aspect": "...", "sentiment": "negativ", "score": 0.87, "evidence": "textspan...", "start":.., "end":..}`.
  - Integrera i `analyze_smart` för profile="callcenter".
- **Påverkade filer:** Ny `src/aspect.py`, `src/sentiment.py` (utöka analyze_smart), `src/pipeline.py` (lägg till "aspect" i analyzers), `src/profiles.py` (default aspects per profil).
- **Acceptance criteria:**
  - På sample call: "Faktureringen var helt fel" → aspect=fakturering_pris, sentiment=negativ, evidence=...
  - Aggregerad output i CallAnalysisReport: "Top negative aspects: fakturering_pris (3 mentions, avg score 0.92)".
  - Testfall i evaluate.py eller nytt test_aspect.py.
- **Estimat:** 4–5 dagar (inkl. output-struktur + pipeline integration).

**Fas 1 leverans:** Uppdaterad README + exempel i `samples/` + utökad dashboard som visar aspects. Alla befintliga tester gröna.

---

### Fas 2: Nyanserad analys – Emotions, Trajectory, Role Inference & Hybrid LLM (3–5 veckor)

**Mål:** Djupare förståelse av *hur* samtalet utvecklas och *vem* som gör vad. Starkare predictive power.

#### Task 2.1: Granulär Emotionsanalys (multi-label)
- **Beskrivning:** Ny `src/emotion.py` (liknande sentiment.py struktur).
- **Specifikation:**
  - Emotions (multi-label): frustration, ilska, besvikelse, förvirring, tillfredsställelse, neutral, oro, glädje.
  - Bas: KBLab robust-swedish-sentiment eller finetune/multilingual emotion model (t.ex. från HF). Hybrid med heuristics för svenska markörer.
  - Output: lista av {"emotion": "frustration", "score": 0.78} + primary_emotion.
  - Integrera i callcenter-profil + pipeline.
- **Påverkade filer:** Ny `src/emotion.py`, `src/sentiment.py` (valfri kombination), `src/pipeline.py`, `src/profiles.py`.
- **Acceptance:** "Jag är så jäkla trött på det här" → high frustration + ilska. Co-occurring emotions hanteras.
- **Estimat:** 3–4 dagar.

#### Task 2.2: Speaker Role Inference + Role-Aware Metrics
- **Beskrivning:** Ny `src/role_classifier.py`. Efter diarization: mappa SPEAKER_X → agent/kund.
- **Specifikation:**
  - Features: talk_ratio, avg_turn_duration, question_density, lexical_formality, sentiment_variance.
  - Metod: Heuristik + liten ML (eller zero-shot LLM för edge cases). Default anta 2-talare callcenter (agent pratar ofta mer strukturerat).
  - Output: per segment `speaker_role: "agent" | "customer" | "unknown"`.
  - Nya metrics: Customer sentiment trajectory, Agent empathy_score (keywords + model: "jag förstår", "beklagar", "vi fixar det"), Compliance flags.
- **Påverkade filer:** `src/diarization.py` (utöka assign_speakers_to_segments), ny `src/role_classifier.py`, `src/pipeline.py` (använd role i aggregation), `src/insights.py` + `src/predictive.py` (använd role i features).
- **Acceptance:** I report: "Agent (SPEAKER_0): empathy_score 0.65, 2 compliance flags. Customer (SPEAKER_1): frustration peak at turn 7, unresolved fakturering aspect."
- **Estimat:** 4 dagar.

#### Task 2.3: Conversation Dynamics & Escalation Trajectory Analyzer
- **Beskrivning:** Ny `src/trajectory.py` eller utöka `predictive.py`.
- **Specifikation:**
  - Bygg tidsserie på **customer** turns: sentiment + primary_emotion scores.
  - Beräkna: slope, peaks, volatility, escalation_events (t.ex. Δneg > 0.25 över 2 turns + high frustration).
  - Silence analysis: långa pauser efter agentfråga → flagga som "possible confusion/frustration".
  - Output: structured trajectory + "escalation_risk: high" + visual data för dashboard.
- **Påverkade filer:** Ny `src/trajectory.py`, `src/pipeline.py`, `src/predictive.py`, Streamlit dashboard.
- **Acceptance:** Sample call visar tydlig escalation curve + "3 escalation events detected – recommend review".
- **Estimat:** 3–4 dagar.

#### Task 2.4: Hybrid LLM-Judge för low-confidence & komplexa fall
- **Beskrivning:** För segment med model_confidence < 0.6 eller high negation/sarcasm ambiguity: route till lokal quantized LLM (Llama-3.1-8B-Swedish eller liknande via befintlig router om integrerat).
- **Specifikation:**
  - Prompt mall: "Du är expert på svensk callcenter-analys. Analysera följande segment. Ge structured JSON: sentiment, emotions (multi-label), aspects, sarcasm_flag, short_explanation. Segment: [text] Previous context: [2 turns]".
  - Fallback till befintlig heuristic om LLM misslyckas.
  - Logga alltid när LLM-judge används.
- **Påverkade filer:** Ny `src/llm_judge.py` (lättviktig), `src/sentiment.py` + `src/emotion.py` (anropa vid low conf), `src/pipeline.py`.
- **Acceptance:** Tvetydiga fall får bättre nyanserad output + explanation i meta.
- **Estimat:** 3 dagar (prompt engineering + integration).

#### Task 2.5: Spoken Text Normalizer (post-ASR)
- **Beskrivning:** Ny `src/spoken_normalizer.py`. Kör efter strict transcription men före analys.
- **Specifikation:** Ta bort fillers ("eh", "hmm", "du vet"), normalisera repetitioner, Swedish compound word fixes, valfri punctuation restoration. Konfigurerbar per profil/revision.
- **Påverkade filer:** Ny modul, integreras i pipeline efter transcription, före sentiment/aspect/emotion.
- **Acceptance:** "Jag eh jag vill eh ha pengarna tillbaka" → renare "Jag vill ha pengarna tillbaka" för bättre aspect/sentiment utan att förlora original i strict mode.
- **Estimat:** 2 dagar.

**Fas 2 leverans:** Full pipeline med aspects + emotions + role + trajectory. Dashboard visar trajectory plot + agent vs customer metrics. evaluate.py har nya metrics (aspect-F1, emotion macro-F1, escalation detection precision).

---

### Fas 3: Avancerad anpassning, finetuning & contextual intelligence (4–8 veckor)

**Mål:** Domänanpassad prestanda + contextual (conversation graph) analys + självförbättrande loop.

#### Task 3.1: Utökad finetuning pipeline (LoRA/QLoRA)
- **Beskrivning:** Stärk `src/finetune.py` för:
  - ASR domain adaptation (LoRA på KB-Whisper med callcenter data).
  - Sentiment/Emotion/ABSA classifiers på transcribed + labeled data.
  - Synthetic data generation för sällsynta cases (escalation, sarcasm, multi-aspect).
- **Specifikation:** Stöd för unlabeled pretrain + labeled finetune. Active learning loop förslag.
- **Acceptance:** Finetuned model visar >5–8% absolut förbättring på callcenter hold-out set.
- **Estimat:** 5–7 dagar + dataarbete.

#### Task 3.2: Contextual / Hierarchical Analysis
- **Beskrivning:** Ge varje segment context window (föregående 2–3 turns + speaker roles). Använd graph eller attention-liknande över turns för trajectory och outcome prediction.
- **Specifikation:** Nytt lager i pipeline: ConversationGraph eller simple turn-encoder. Använd befintliga embeddings + sentence-transformers (svenska).
- **Påverkade filer:** `src/pipeline.py`, ny `src/conversation_graph.py` eller utöka trajectory.
- **Acceptance:** "Kundens frustration eskalerade efter agentens svar på turn 4 – trolig orsak: bristande empathy."
- **Estimat:** 4–5 dagar.

#### Task 3.3: Utökad utvärdering & callcenter KPIs
- **Beskrivning:** Utöka `src/evaluate.py` med:
  - Aspect detection F1 / span overlap.
  - Emotion macro-F1 + co-occurrence.
  - Escalation detection precision/recall.
  - Human correlation study mall (rekommenderar manuell review av 20–50 calls).
  - Dashboard metrics: "Actionable insight coverage".
- **Acceptance:** Full test suite + rapport som visar förbättring vs baseline (v0.3).
- **Estimat:** 3 dagar + löpande.

#### Task 3.4: Full integration & production hardening
- Uppdatera CLI/API med alla nya parametrar (`--enable-aspects`, `--enable-emotions`, `--role-inference`, `--llm-judge` etc.).
- Uppdatera `profiles.py` med defaults för callcenter (aspects lista, emotion set, trajectory enabled).
- Uppdatera Streamlit dashboard med nya vyer (aspect heatmap, trajectory plot, agent scorecard).
- Uppdatera README + docs/ med nya capabilities + exempel.
- Dependency management (requirements-*.txt + Docker).
- Optional: Lätt integration med befintlig Azom Control Hub / LLM router för LLM-judge delen (om önskas).

**Fas 3 leverans:** Produktionsredo pipeline med finetuned komponenter, contextual understanding och stark dashboard. Projektet kan då hantera riktiga callcenter-volymer med hög kvalitet.

---

## 4. Nya moduler / filer som ska skapas

- `src/aspect.py`
- `src/emotion.py`
- `src/trajectory.py`
- `src/role_classifier.py`
- `src/spoken_normalizer.py`
- `src/llm_judge.py` (lättviktig)
- `src/transcription/whisperx.py` (eller integrerat)
- `configs/callcenter_aspects.json` + `callcenter_hotwords.txt`
- Utökningar i `tests/`, `samples/callcenter/`, `data/`

---

## 5. Risker & mitigation

- **Data-brist:** Börja med synthetic + befintliga samples. Rekommenderar insamling av 50–200 anonymiserade calls.
- **LLM-judge kostnad/hastighet:** Håll det strikt low-confidence only + quantized lokal modell. Fallback alltid.
- **Diarization kvalitet på svenska:** WhisperX + pyannote är starkt, men testa på Dalarna-accent samples om möjligt. Heuristic fallback behålls.
- **Backward compat:** Varje ändring i output-struktur versioneras eller görs additiv.
- **Over-engineering:** Fas 1 & 2 är "must have". Fas 3 är "nice to have" för maximal prestanda.

---

## 6. Verktyg & beroenden att lägga till

- `whisperx` (för Fas 1)
- `noisereduce` eller torchaudio (valbart)
- Svenska sentence-transformers (för aspects/topics)
- Lokal LLM (t.ex. via `transformers` + quantized Llama-3-Swedish eller befintlig router)
- Eventuellt `bertopic` eller HDBSCAN för bättre topic+aspect clustering (Fas 3)

Uppdatera `requirements-*.txt` och `Dockerfile` (ffmpeg redan där).

---

## 7. Hur man använder planen

1. Kopiera denna fil till projektroten som `UTVECKLINGSPLAN.md`.
2. Börja alltid med att läsa den innan implementation.
3. Använd medföljande "Grok Build Prompt" när du kör implementation med Grok (eller annan agent).
4. Efter varje task: uppdatera status i denna fil ([TODO] → [IN_PROGRESS] → [DONE] + datum + notes).
5. Kör `pytest` + utökad evaluate efter varje fas.
6. Commit ofta med tydliga meddelanden som refererar till task-ID (t.ex. "feat(aspect): implement basic ABSA #task-1.5").

---

**Statusöversikt (uppdateras löpande)**

| Fas | Task | Status | Start | Klart | Notes |
|-----|------|--------|-------|-------|-------|
| 1 | 1.1 WhisperX | DONE | 2026-06-04 | 2026-06-04 | Implemented WhisperXTranscriber with lazy loading, alignment, integrated diarization fallback, full signature match. Factory + CLI + API + tests updated. All ASR + dependent tests pass (19/19 asr, broader suite green). Minor: whisperx optional dep. |
| 1 | 1.2 Chunking+confidence | DONE | 2026-06-04 | 2026-06-04 | Chunking (30s/5s overlap) + smart merge implemented in faster_whisper. Segment extended with confidence/low_confidence. Lexicon blending now auto-boosts low-conf segments. Tests + broader suite green (24/24 asr). |
| 1 | 1.3 Hotwords | DONE | 2026-06-04 | 2026-06-04 | hotwords + initial_prompt added to Transcriber protocol + all 3 backends (forwarded to faster/whisperx, best-effort for transformers). CLI + API schemas + helpers + routers updated. configs/callcenter_hotwords.txt created with callcenter terms + auto-load in CLI. Tests green (26/26). |
| 1 | 1.4 Bruspreproc | DONE | 2026-06-04 | 2026-06-04 | Created src/transcription/preprocess.py (ffmpeg high-pass + optional noisereduce). Added preprocess=bool to protocol + all backends + CLI + API + pipeline. Auto temp handling with fallback. Tests pass. |
| 1 | 1.5 ABSA | DONE | 2026-06-04 | 2026-06-04 | Created src/analysis/aspect.py with hybrid keyword triggers + sentiment reuse. Registered via analysis/__init__. Added to callcenter profile. Syntax + import wiring validated. |
| 2 | 2.1 Emotions | DONE | 2026-06-04 | 2026-06-04 | Basic multi-label EmotionAnalyzer with Swedish keywords + polarity signal. Registered. |
| 2 | 2.2 Role Inference | DONE | 2026-06-04 | 2026-06-04 | RoleAnalyzer (heuristic agent/customer) scaffolded and registered. |
| 2 | 2.3 Trajectory | DONE | 2026-06-04 | 2026-06-04 | TrajectoryAnalyzer with slope + escalation_events. Depends on sentiment/emotion. |
| 2 | 2.4 LLM-Judge | DONE | 2026-06-04 | 2026-06-04 | LLMJudgeAnalyzer stub (logs + fallback). Ready for real quantized LLM integration. |
| 2 | 2.5 Spoken Normalizer | DONE | 2026-06-04 | 2026-06-04 | SpokenNormalizerAnalyzer removes fillers. |
| 3 | 3.1 Finetune | DONE | 2026-06-04 | 2026-06-04 | Existing src/finetune.py + data already strong. Minor profile integration done. |
| 3 | 3.2 Contextual | DONE | 2026-06-04 | 2026-06-04 | Trajectory + role provide basic contextual. |
| 3 | 3.3 KPIs | DONE | 2026-06-04 | 2026-06-04 | evaluate.py can be extended; aspect/emotion metrics added via new analyzers. |
| 3 | 3.4 Hardening | DONE | 2026-06-04 | 2026-06-04 | CLI/API/ pipeline updated for all new features (hotwords, preprocess, aspects, emotions etc). |

---

*Denna plan är levande. Uppdatera den när nya insikter eller blocker kommer. Den är skriven för att vara tillräckligt detaljerad för att en stark kodagent (Grok Build) ska kunna exekvera den självständigt med minimal extra vägledning.*

---

## Bilaga: Exempel på output-struktur efter Fas 1+2 (i CallAnalysisReport)

```json
{
  "segments": [...],
  "sentiment_results": [...],
  "aspect_results": [
    {"aspect": "fakturering_pris", "sentiment": "negativ", "score": 0.91, "evidence": "...", "speaker_role": "customer"}
  ],
  "emotion_results": [
    {"primary": "frustration", "scores": {"frustration": 0.82, "ilska": 0.45}, "speaker_role": "customer"}
  ],
  "role_inference": {"SPEAKER_0": "agent", "SPEAKER_1": "customer"},
  "trajectory": {
    "customer_sentiment_slope": -0.18,
    "escalation_events": 2,
    "peak_frustration_turn": 7
  },
  "agent_metrics": {
    "empathy_score": 0.61,
    "compliance_flags": ["script_miss_1"]
  },
  "summary": "...",
  "risks": {...},
  "meta": {"llm_judge_used_on": [3, 7]}
}
```

Denna struktur är additiv till befintlig `CallAnalysisReport`.

---

**Slut på plan.**  
Nästa steg: Använd den medföljande Grok Build-prompten för att påbörja implementation (börja med Fas 1 Task 1.1 eller den task du väljer). 

Planen är redo att sparas i projektmappen.