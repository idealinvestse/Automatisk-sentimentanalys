# Fas 3: Mistral/OpenRouter LLM Integration

**European-first hybrid LLM layer for holistic call center conversation intelligence.**

This document is the practical quickstart and reference for the Mistral integration (see `UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md` for the full plan and status).

## Why this layer?

Local per-segment models (XLM-R sentiment, heuristics, trajectory) are fast, cheap, offline, and GDPR-friendly for most work.

They are insufficient for:
- Cross-turn causality and sarcasm
- True root cause (beyond the first complaint)
- High-quality, coachable "actionable summary" and agent assessment with evidence

**Mistral Medium 3.5** (primary: `mistralai/mistral-medium-3.5`) and Large 3 via OpenRouter deliver exactly that holistisk view when needed.

**Design principles (non-negotiable):**
- **Hybrid always**: Local = base / fast path. Mistral = selective deep path.
- **Strict structured**: `json_schema` + `strict: true` + Pydantic validation.
- **Privacy first**: Every external call is logged with clear "data sent to third-party" notice. Caching is mandatory. Fallback is automatic.
- **Profile driven + explicit flags**: `callcenter` enables by default (selectively). CLI/API/code can force/skip.
- **Cost control**: Disk cache by content hash. Per-call budget warning. Meta always contains `cost_usd`.

## Quickstart (< 30 minutes for a new user)

### 1. Prerequisites
```powershell
# Install (openai is required only for the LLM path)
pip install -r requirements.txt
# or
pip install openai pydantic
```

Set your key (get one at openrouter.ai):
```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

### 2. Activate via CLI (easiest)
```powershell
python -m src.cli analyze-call your_call.wav `
  --backend whisperx --diarize `
  --use-mistral-llm `
  --llm-model mistralai/mistral-medium-3.5
```

You will see:
- Yellow warning: "Mistral/OpenRouter LLM deep analysis ENABLED..."
- INFO log with the full GDPR egress notice.
- In the final report: `llm` key with `actionable_summary`, `agent_assessment`, `trajectory`, `root_cause`, plus `meta` (model, cost, cached, etc.).

### 3. Activate via Python / Pipeline
```python
from src.pipeline import CallAnalysisPipeline

pipe = CallAnalysisPipeline(
    profile="callcenter",           # auto-enables for callcenter (selective)
    # or force:
    use_mistral_llm=True,
    llm_model="mistralai/mistral-medium-3.5",  # or mistralai/mistral-large-3
    deep_analysis=True,             # stronger signal
)

report = pipe.analyze_audio("call.wav", run_diarization=True)
print(report.llm)                   # the Mistral output
print(report.results.get("llm"))    # also here
```

`report.to_dict()` includes `"llm": {...}` (additive – all previous fields are unchanged).

### 4. Via API
```json
POST /analyze_pipeline
{
  "segments": [ {"text": "...", "speaker": "SPEAKER_0"}, ... ],
  "use_mistral_llm": true,
  "llm_model": "mistralai/mistral-medium-3.5",
  "deep_analysis": false
}
```

Response includes `"llm": { ... }`.

### 5. Profile control (no code change)
`callcenter` profile (and aliases) has:
```python
"llm": {
    "enabled": True,
    "default_model": "mistralai/mistral-medium-3.5",
    "cost_budget_per_call": 0.08,
    ...
}
```

See `src/profiles.py` and `configs/llm_config.yaml` (example).

### 6. Dashboard
```powershell
streamlit run app/dashboard.py
```
- Sidebar → "Mistral LLM" checkbox.
- Live-analys section will show **✨ LLM-enhanced (Mistral via OpenRouter)** badge + nice expanders for:
  - Actionable Summary (QA recommendations)
  - Agent Assessment (empathy score + flags + evidence)
  - Trajectory (kundresa + escalation)
  - Root Cause

## Privacy & GDPR (must-read)

Every time data leaves the system:

```
INFO ... EXTERNAL LLM CALL (OpenRouter/Mistral) | model=... | task=... | chars≈... | 
Data (full conversation transcript + roles) is being sent to a third-party service. 
This is only done when the callcenter profile / --use-mistral-llm / deep_analysis enables it.
```

- Only happens on explicit enablement.
- Transcript is role-labeled but otherwise raw (PII redaction hook prepared in profile `anonymize_before_llm` for future Fas 3.4 work).
- Caching means the same call is never sent twice.
- `meta` always tells you exactly what model + cost + whether it was cached.
- Fallback on any error (auth, rate limit, timeout, bad JSON) – you still get the full local report.

**Recommendation**: For production with real customer calls, consider the `anonymize_before_llm` flag + a redactor before enabling at scale.

## Output Structure (example)

```json
{
  "llm": {
    "trajectory": {
      "summary": "Kundens frustration eskalerade efter tur 4 p.g.a. bristande bekräftelse av fakturaproblemet.",
      "customer_sentiment_slope": -0.18,
      "escalation_events": ["Tur 4: 'Det här är helt galet!' (CUSTOMER)"]
    },
    "root_cause": {
      "primary_cause": "Agent missade att validera kundens upplevelse tidigt → eskalerade till krav på kompensation.",
      "customer_unresolved": true
    },
    "actionable_summary": {
      "problem": "Kunden fick felaktig faktura och upplevde ingen hjälp.",
      "recommendations_for_qa": [
        "Säg 'Jag hör att det här är frustrerande för dig' direkt efter kunden nämner fakturan.",
        "Erbjud konkret nästa steg inom 30 sekunder."
      ],
      "risk_level": "high"
    },
    "agent_assessment": {
      "empathy_score": 0.42,
      "compliance_flags": ["missed_empathy_early"],
      "evidence_spans": [{"text": "...", "speaker_role": "agent", "turn_index": 3}]
    },
    "meta": {
      "model": "mistralai/mistral-medium-3.5",
      "cost_usd": 0.023,
      "cached": false,
      "llm_used": true,
      "latency_s": 4.2
    }
  }
}
```

Full schemas: `src/llm/schemas.py` (Pydantic v2, `extra='forbid'`, descriptions included in JSON schema sent to the model).

## Caching & Cost

- Cache location: `.cache/llm/<sha256>.json` (gitignored via `cache/` rule).
- Key includes: model + full messages (transcript + roles + local context) + task + schema name.
- Second identical run on same input → `cached: true`, `cost_usd: 0.0`.
- `python -m src.llm.openrouter_client` has no CLI, but you can call `client.clear_cache()` from code for testing/privacy.
- Budget warning logged if `cost_budget` is set on the client and exceeded.

See `src/llm/openrouter_client.py` (PRICING table, `_make_cache_key`, `_save_to_cache` etc.).

## Evaluation

```powershell
python -m src.evaluate llm-quality --output reports/llm_quality.json
```

Produces:
- `fallback_rate`
- `avg_cost_usd`, `total_cost`
- `pct_with_actionable`, `pct_with_evidence`
- Consistency note (second run should be cached)

For real human preference on insights quality, run on a labeled set of 20–50 calls (recommended in the plan).

## Fallback & Error Handling

If OpenRouter fails (no key, 401, rate limit after retries, timeout, JSON parse error despite strict mode):
- Analyzer returns `{"fallback": true, "error": "...", "meta": {"llm_used": false, "llm_fallback_reason": "..."}}`
- Pipeline still returns a complete `CallAnalysisReport` with all local results.
- You never lose the fast-path data.

## Files & Extension Points

- `src/llm/` – all new code (do not mix with per-segment analyzers in `src/analysis/`).
- Pipeline integration only in `src/pipeline.py` ( `_should_use_mistral_llm`, `_run_mistral_holistic`).
- Prompts can be A/B tested by subclassing or replacing `get_system_prompt()` / `build_user_prompt()`.
- Future: `src/llm/pii_redactor.py` behind the `anonymize_before_llm` profile flag.

## Common Pitfalls & Tips

- No `OPENROUTER_API_KEY` → every call falls back (you will see the warning on client creation).
- Very short calls (< 5–6 turns) rarely benefit – the heuristics in pipeline skip unless you force with the flag.
- Long transcripts + high `max_tokens` cost money – rely on cache during development.
- Always look at `report.llm["meta"]` before trusting the insights.

## References

- Full plan & task status: `UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md`
- Architecture updates: `docs/ARCHITECTURE.md`
- Config example: `configs/llm_config.yaml`
- Schemas & prompts: `src/llm/`
- Smoke LLM eval report: `reports/llm_quality_smoke.json` (after running evaluate)

New users should be able to go from "never heard of the project" to "I have LLM-enhanced insights on my first call" in well under 30 minutes following the Quickstart above.

---

*This integration prioritizes European models, strict output contracts, aggressive caching, and transparent external data handling so that call center intelligence can be both powerful and responsible.*
