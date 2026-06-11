# LLM Agent Quick Reference

**Minimal context version** for coding agents. Read `docs/LLM_AGENT_GUIDE.md` for full details.

## Core Philosophy
- Hybrid-first (local + heuristics default, Mistral selective)
- Graceful degradation mandatory
- Privacy by design + explicit "EXTERNAL LLM CALL" logging
- Registry-based analyzers

## Key Files

| Purpose                    | File                              |
|----------------------------|-----------------------------------|
| Main orchestration         | `src/pipeline.py`                 |
| Add new analysis logic     | `src/analysis/registry.py`        |
| ASR backends               | `src/transcription/factory.py`    |
| LLM (Mistral)              | `src/llm/mistral_analyzer.py`     |
| API entrypoint             | `src/api/app.py`                  |
| CLI                        | `src/cli.py`                      |
| Current status             | `docs/ROADMAP.md`                 |
| Full agent guide           | `docs/LLM_AGENT_GUIDE.md`         |

## How to Add a New Analyzer (Preferred Pattern)

1. Create `src/analysis/your_analyzer.py` inheriting `BaseAnalyzer`
2. Implement `analyze(self, context)`
3. Register in `registry.py`
4. Add test

## Pipeline Order (non-fatal per step)
Transcription → PII Redaction → Registry Analyzers → Agent Performance → Optional LLM → QA → Alerting

## Important Rules
- Never remove fallback logic
- Never send transcripts to LLM without explicit flag
- Always run `ruff format && ruff check && pytest` before PR
- Respect `callcenter` profile for PII redaction

## Common Commands
```bash
pip install -e ".[dev,diarize]"
pytest -q
python -m src.cli analyze-call samples/call.wav --backend faster --language sv
uvicorn src.api:app --reload
python -m src.evaluate llm-quality
```

**When in doubt**: Follow patterns in existing analyzers and `pipeline.py`.