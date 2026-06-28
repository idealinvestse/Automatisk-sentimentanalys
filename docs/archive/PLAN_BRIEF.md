# LLM-Judge Core Implementation - Plan Brief (FÖRSLAG A)

**Scope:** SMALL FOCUSED — only core LLM-judge (confidence routing + budget + fallback). No dashboard, no pipeline wiring.

**Branch:** feat/v0.5-llm-judge (based on main@c63078d)

**3-Phase Execution:**
1. **A.1** — Extend src/llm/schemas.py with `LLMJudgeVerdict` + `LLMJudgeResult` (exact Pydantic models from spec)
2. **A.2** — Replace stub in src/analysis/llm_judge.py: confidence threshold (default 0.6), max_segments_per_call, budget $0.10, batch≤5, provider (openrouter/groq), model llama-3.1-8b-instant, graceful fallback, EXTERNAL LLM CALL log, requires=["sentiment"]
3. **A.3** — Create tests/test_llm_judge.py with 8+ mocked tests (no real API)

**Rules enforced:**
- Read full files before edit
- No new dependencies (pydantic/openai/httpx already present)
- Graceful degradation on all LLM failures
- Google-style docstrings
- Commit + push after each phase
- Ruff clean at end

**Failure handling:** torch not required; static verify if needed.

**Deliverable:** 3 commits, branch pushed, test report + ruff, final summary to main session.