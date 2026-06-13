# Backend + API Testing Status

**Status: SIGN-OFF 2026-06-13** — Backend/API refactor Fas 0–5 complete.

## Completed phases

### Fas 0 — Audit
Baseline: ruff/mypy/pytest inventory, architecture gaps documented.

### Fas 1 — Hardening core
- Structured errors (`error_code` + `request_id` in JSON body)
- `scan_process` → `run_route` + `asyncio.to_thread`
- `AsrParamsMixin` + `asr_kwargs_from` (hotwords/initial_prompt wired)

### Fas 2 — Service layer
- `src/api/services/` (`conversation`, `pipeline_cache`)
- Fas 4 cache-before-reanalyze (`resolve_reports`, `reanalyze` flag)
- Optional `use_full_pipeline` on `/analyze_conversation`

### Fas 3 — Production hardening
- Rate limiting (`API_RATE_LIMIT_RPM`, `rate_limit_exceeded` error code)
- `sentiment_profile` default `callcenter` on conversation/scan/batch
- `use_full_pipeline` on `scan_process` analyze operation
- `asyncio.to_thread` on batch transcribe + pipeline routers
- `agent_id` format validation on Fas 4 agent performance

### Fas 4 — Documentation
- `docs/API.md` updated (errors, rate limit, full pipeline, reanalyze)

### Fas 5 — Validation
- `tests/test_scan_logic.py`
- `tests/contracts/test_api_error_contract.py`
- Full API suite: **74+ tests**, `src/api` coverage **≥ 90%**

## Run verification

```bash
ruff check src/api/
pytest tests/test_api.py tests/test_api_smoke.py tests/test_api_coverage.py \
  tests/test_api_services.py tests/test_scan_logic.py \
  tests/contracts/test_api_error_contract.py \
  --cov=src/api --cov-fail-under=90
```

## Deferred (low ROI / out of scope)
- RFC 7807 Problem Details (structured errors sufficient)
- Load testing / p99 latency benchmarks
- Full mypy clean on transitive pipeline deps