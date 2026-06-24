# LLM Agent Guide for Automatisk-sentimentanalys

**Purpose**: This document is specifically optimized for LLM coding agents (Grok, Claude, GPT-4o, OpenAI Codex, Cursor, Windsurf, etc.). It provides deep project understanding, clear extension patterns, and robust guidelines so agents can contribute effectively, safely, and consistently.

> **Rule for all agents**: Read this file + `docs/ROADMAP.md` + `README.md` before making any code changes.

## 1. Project Mission & Philosophy

**Mission**: Build the best open-source Swedish Call Center Intelligence platform — combining high-accuracy ASR, speaker diarization, sentiment/intent/emotion analysis, agent performance evaluation, QA/compliance scoring, and selective LLM reasoning — while keeping data private and costs low.

**Core Philosophy**:
- **Hybrid-first**: Local models + heuristics are the default/fast/private path. Mistral (via OpenRouter) is used selectively for high-value reasoning.
- **Graceful degradation**: Missing optional components (pyannote, whisperx, etc.) must fall back automatically.
- **Privacy by design**: Explicit logging of external LLM calls. Early PII redaction for callcenter profile.
- **Extensibility**: Registry-based analyzers and clear plugin points.
- **Production realism**: Error isolation, caching, and non-fatal failures where possible.

## 2. High-Level Architecture

```
Audio / Text Input
       ↓
Transcription + Diarization (src/transcription/ + diarization.py)
       ↓
PII Redaction (early, profile-dependent)
       ↓
Analysis Registry (src/analysis/registry.py) → multiple analyzers run in topological order
       ↓
Agent Performance + QA/Compliance scoring
       ↓
Optional Mistral LLM holistic analysis (src/llm/)
       ↓
Alerting + Insights aggregation
       ↓
CallAnalysisReport (returned to CLI / API / Dashboard)
```

**Key Integration Points**:
- `src/pipeline.py`: `CallAnalysisPipeline` orchestrates everything.
- `src/analysis/registry.py`: Central place to register new analyzers.
- `src/transcription/factory.py`: Chooses ASR backend.
- `src/llm/`: Mistral/OpenRouter integration with strict structured output.

## 3. Directory Structure & Responsibilities

| Path                        | Purpose                                                                 | Key Files to Know |
|-----------------------------|-------------------------------------------------------------------------|-------------------|
| `src/`                      | Main source code                                                        | - |
| `src/pipeline.py`           | Core orchestration (`CallAnalysisPipeline`)                             | Most important file |
| `src/analysis/`             | All analyzers (aspect, emotion, role, trajectory, intent, etc.) + registry | `registry.py`, `base.py` |
| `src/transcription/`        | ASR backends (faster_whisper, transformers, whisperx) + preprocess     | `factory.py`, `base.py` |
| `src/llm/`                  | Mistral/OpenRouter + Groq Cloud client, prompts, schemas, analyzers  | `mistral_analyzer.py`, `groq_analyzer.py`, `groq_client.py`, `prompts.py`, `schemas.py` |
| `src/api/`                  | FastAPI application (`app.py` exposes `app`)                            | `app.py`, `routers/`, `schemas.py` |
| `src/cli.py`                | Typer-based CLI (`sentiment`, `transcribe`, `analyze-call`)             | - |
| `src/diarization.py`        | Speaker diarization with pyannote + heuristic fallback                  | - |
| `src/profiles.py`           | Profile resolution (forum, callcenter, news, etc.)                      | - |
| `src/lexicon.py`            | Lexicon blending                                                        | - |
| `configs/`                  | YAML configs, hotwords, llm_config.yaml                                 | `callcenter_hotwords.txt` |
| `docs/`                     | All documentation                                                       | `ROADMAP.md`, `API.md`, this file |
| `tests/`                    | 57 test files (581 test functions)                                    | `test_pipeline.py`, `test_llm_*.py` |

## 4. Getting Started (for Agents)

```bash
# Recommended setup for agents
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,diarize]"

# Run tests
pytest -q

# Run CLI example
python -m src.cli analyze-call samples/call.wav --backend faster --language sv --log-level INFO

# Start API
uvicorn src.api:app --reload
```

**Important environment variables**:
- `OPENROUTER_API_KEY` (for LLM features)
- `HF_TOKEN` (for pyannote models)

## 5. Core Patterns & Conventions

### 5.1 Analyzer Registry Pattern (Preferred way to add new analysis)

All new analysis logic should be added as analyzers in `src/analysis/`:

1. Create `src/analysis/your_analyzer.py`
2. Inherit from `BaseAnalyzer` (see `base.py`)
3. Implement `analyze(self, context: AnalysisContext) -> dict`
4. Register it in `registry.py` with a unique name and dependencies.

**Example skeleton**:
```python
# src/analysis/your_analyzer.py
from .base import BaseAnalyzer

class YourAnalyzer(BaseAnalyzer):
    name = "your_analyzer"
    depends_on = ["sentiment", "role"]   # topological order

    def analyze(self, context):
        # context.segments, context.transcript, etc.
        return {"your_key": result}
```

Then register:
```python
# src/analysis/registry.py
REGISTRY.register("your_analyzer", YourAnalyzer)
```

### 5.2 Pipeline Steps (in CallAnalysisPipeline)

The pipeline runs in this order (non-fatal errors are caught per step):
1. Transcription + Diarization
2. PII Redaction
3. Text Analysis via Registry
4. Agent Performance
5. Optional Mistral LLM (`_should_use_mistral_llm`)
6. QA / Compliance
7. Alerting

When modifying pipeline logic, keep error isolation (`try/except` + logging + continue).

### 5.3 LLM Integration (`src/llm/`)

- Two providers: **Mistral/OpenRouter** (default, EU/GDPR) and **Groq Cloud** (fast/cheap, US/Saudi-hosted).
- Use `provider=` flag (`"openrouter"` | `"groq"`) to select.
- Use strict structured output with Pydantic + `response_format`.
- Always log `"EXTERNAL LLM CALL (Groq/Mistral)"` when calling external APIs.
- Cache results in `.cache/llm/`.
- Fallback to local analysis on any LLM failure.
- GDPR gate: Groq requires `groq_eu_residency=True` or `anonymize_before_llm=True`.
- Prompts live in `prompts.py`. Schemas in `schemas.py`.
- See `docs/LLM_PROVIDERS.md` for full comparison matrix.

### 5.4 Graceful Degradation

- If `pyannote.audio` is missing → use heuristic VAD in `diarization.py`.
- If LLM key is missing → skip Mistral step.
- If a single analyzer fails → log warning and continue.

**Never** let missing optional dependencies crash the whole pipeline.

## 6. How to Extend the System (Agent Playbooks)

### Add a new Analyzer
1. Follow pattern in section 5.1.
2. Add tests in `tests/test_analysis_registry.py` or new test file.
3. Update `docs/ROADMAP.md` if it's a major feature.

### Add a new ASR Backend
1. Create file in `src/transcription/your_backend.py`.
2. Inherit from `TranscriptionBackend`.
3. Register in `factory.py`.
4. Add to optional dependencies in `pyproject.toml` if heavy.

### Modify LLM Behavior
- Edit prompts in `src/llm/prompts.py`.
- Update Pydantic schemas in `src/llm/schemas.py`.
- Test with `python -m src.evaluate llm-quality`.

### Add new API endpoint
1. Add router in `src/api/routers/`.
2. Update `src/api/app.py` if needed.
3. Add schema in `src/api/schemas.py`.
4. Document in `docs/API.md`.
5. Add test in `tests/test_api*.py`.

### Modify CLI
- Main file: `src/cli.py` (Typer commands).
- Keep rich output and progress bars consistent.

## 7. Coding Standards (Mandatory for Agents)

- **Formatting & Linting**: Run `ruff format` and `ruff check` before committing.
- **Type hints**: Use them. Run `mypy src`.
- **Error handling**: Prefer explicit try/except with logging over silent failures (except where graceful degradation is intended).
- **Logging**: Use `logging.getLogger(__name__)`.
- **No hardcoded secrets**: Use environment variables or `src/core/config.py`.
- **Docstrings**: Add Google-style or NumPy-style docstrings on public functions/classes.
- **Tests**: New features must have tests. Aim for high coverage on `src/api/` and `src/pipeline.py`.

## 8. Security & Privacy Rules (Critical)

- Never commit real customer audio or transcripts.
- Always respect `SENTIMENT_API_KEY` and `OPENROUTER_API_KEY`.
- PII redaction must run early for `callcenter` profile.
- External LLM calls must be explicitly logged.
- See full rules in `SECURITY.md`.

## 9. Testing Strategy

- Unit tests: `tests/test_*.py`
- Pipeline tests: `tests/test_pipeline.py`
- API tests: `tests/test_api*.py` (target ≥90% coverage)
- LLM quality: `python -m src.evaluate llm-quality`
- Run full suite: `pytest`

## 10. Quick Reference for Agents

| Task                              | Primary File(s)                          | Key Command / Pattern |
|-----------------------------------|------------------------------------------|-----------------------|
| Understand current capabilities   | `docs/ROADMAP.md`                        | - |
| Run full analysis                 | `src/cli.py`                             | `analyze-call` |
| Add new analysis logic            | `src/analysis/registry.py` + new file    | Registry pattern |
| Call Mistral                      | `src/llm/mistral_analyzer.py`            | Structured output |
| Start API                         | `src/api/app.py`                         | `uvicorn src.api:app` |
| Check dependencies                | `pyproject.toml`                         | optional-dependencies |
| Update documentation              | `docs/ROADMAP.md`, `README.md`           | - |

## 11. What NOT to Do

- Do not remove graceful fallback logic.
- Do not send transcripts to LLM without explicit user flag or profile setting.
- Do not hardcode paths or API keys.
- Do not bypass the analyzer registry for new analysis features.
- Do not ignore existing tests when modifying core files.

---

**End of LLM Agent Guide**. 

This document should be the first thing any coding agent reads when working on this repository. Update it when architecture or major patterns change.