# API Findings — Phase 1 Code Review

**Branch:** `api-review-v2-hardening`  
**Date:** 2026-06-03  
**Source:** DesignAgent + ImplementationAgents + TestAgent + baseline tooling  
**Plan:** [API_REVIEW_HARDENING_PLAN.md](./archive/API_REVIEW_HARDENING_PLAN.md)

---

## P0 — Critical (fix before production)

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| P0-1 | `deep_analysis` wired to `use_mistral_llm` on `/agent_performance` | `pipeline.py:82` | **FIXED** (Fas 1) |
| P0-2 | No authentication on mutating endpoints | `app.py`, all routers | **FIXED when prod** — set `API_PRODUCTION` or `API_REQUIRE_AUTH` + `SENTIMENT_API_KEY` |
| P0-3 | `llm_api_key` accepted in JSON body | `schemas.py` Fas 4 models | **FIXED when prod** — body ignored unless `API_ALLOW_CLIENT_LLM_KEY=true`; prefer header |
| P0-4 | Server filesystem access via `audio_path` / `directory` | `path_validation.py` | **FIXED when prod** — set `API_PRODUCTION` or `API_REQUIRE_MEDIA_ROOT` + `API_MEDIA_ROOT` |
| P0-5 | `detail=str(e)` leaks internals | routers | **FIXED** — `router_errors.run_route` (Fas 2) |
| P0-6 | No payload limits (`segments_list`, `segments`) | `schemas.py` | **FIXED** — max 50 calls × 200 segments (Fas 2) |
| P0-7 | `cached` flag heuristic incorrect | `pipeline.py:87-88` | **FIXED** — `cache_hit` from `precompute_and_cache` (Fas 2) |

---

## P1 — High (Phase 2 core)

| ID | Issue | Location |
|----|-------|----------|
| P1-1 | No DI; new `CallAnalysisPipeline` + cache per request | `pipeline.py` | **FIXED** — shared cache + `create_pipeline` (Fas 2) |
| P1-2 | Routers catch `Exception` → bypass `app.py` domain handlers | All routers | **FIXED** — `run_route` + `run_route_sync` on alerting (v0.5) |
| P1-3 | Fas 4 always runs full `analyze_segments` before cache | `pipeline.py` | **FIXED** — `resolve_reports` (Fas 2) |
| P1-4 | Missing `deep_analysis` / `llm_model` on Fas 4 requests | `schemas.py` | **FIXED** — `Fas4LlmFlags` (Fas 2) |
| P1-5 | `getattr(req, 'llm_api_key')` redundant | `pipeline.py` | **FIXED** |
| P1-6 | No CORS, rate limit, security headers, request ID | `app.py` | **FIXED** — rate limit `API_RATE_LIMIT_RPM` (Fas 3) |
| P1-7 | `profile="call"` vs `callcenter` in conversation/scan | `conversation.py`, `scan.py` | **FIXED** — `sentiment_profile` default `callcenter` (Fas 3) |
| P1-8 | Path/body `agent_id` not validated | `pipeline.py:71-72` | **FIXED** — regex validator (Fas 3) |
| P1-9 | Blocking work in `async def` handlers | All routers | **FIXED** — `asyncio.to_thread` on heavy paths (Fas 1–3) |
| P1-10 | `LLMError` not registered in `app.py` | `app.py` | **FIXED** (Fas 1) |
| P1-11 | `AlertsRequest` allows empty inputs | `schemas.py:425-429` | **FIXED** — `require_input` validator |
| P1-12 | mypy errors in `conversation.py`, `pipeline.py` | See baseline |

---

## P2 — Medium

| ID | Issue | Location |
|----|-------|----------|
| P2-1 | API version `0.3.0` vs plan target | `app.py:47` |
| P2-2 | Flat `{"detail"}` vs RFC 7807 | `app.py` | **PARTIAL** — `detail` + `error_code` + `request_id` (Fas 1); RFC 7807 deferred |
| P2-3 | Mid-file imports (ruff E402) | `pipeline.py:62+` |
| P2-4 | Weak typing on segment dicts | `schemas.py` |
| P2-5 | OpenAPI examples sparse on Fas 4 | `schemas.py` |
| P2-6 | Test coverage ~75% (target 90%) | `tests/test_api.py` |

---

## Endpoint inventory

| Method | Path | Request | Response | Auth | Cache | Notes |
|--------|------|---------|----------|------|-------|-------|
| GET | `/health` | — | dict | None | — | Liveness |
| POST | `/analyze` | AnalyzeRequest | AnalyzeResponse | None | sentiment | |
| POST | `/transcribe` | TranscribeRequest | TranscribeResponse | None | ASR | Server paths |
| POST | `/batch_transcribe` | BatchTranscribeRequest | BatchTranscribeResponse | None | — | workers ≤ 8 |
| POST | `/analyze_conversation` | AnalyzeConversationRequest | AnalyzeConversationResponse | None | — | profile=call |
| POST | `/batch_analyze_conversation` | BatchAnalyzeConversationRequest | BatchAnalyzeConversationResponse | None | — | |
| POST | `/analyze_pipeline` | PipelineRequest | PipelineResponse | None | per-request | Has `deep_analysis` |
| POST | `/agent_performance/{id}` | AgentPerformanceRequest | AgentPerformanceResponse | None | aggregate | P0-1 bug |
| POST | `/search/semantic` | SemanticSearchRequest | SemanticSearchResponse | None | — | |
| POST | `/insights/hot_topics` | HotTopicsRequest | HotTopicsResponse | None | hot topics | |
| POST | `/qa/score` | QAScoreRequest | QAScoreResponse | None | — | |
| POST | `/alerts` | AlertsRequest | AlertsResponse | None | — | |
| POST | `/scan_process` | ScanProcessRequest | ScanProcessResponse | None | state file | |

---

## Baseline metrics (Fas 0)

| Metric | Value |
|--------|-------|
| `src/api` coverage (`test_api.py` + `test_api_coverage.py`) | **96.38%** |
| Tests | 53 passed |
| CI | `api-test` job, `--cov-fail-under=90` |
| ruff `src/api` | 1 error (E402 mid-file imports) |
| mypy `src/api` | 2 errors |
| bandit | Skipped (bandit not in venv; install in Fas 5) |
| OpenAPI baseline | `reports/api_openapi_baseline.json` |

---

## Phase 2 implementation order

1. `dependencies.py` + lifespan `app.state`
2. Security: API key auth, secret handling
3. Schema limits + `deep_analysis` on all Fas 4 requests
4. Router DI refactor + exception taxonomy
5. Middleware (CORS, rate limit, headers, request ID)
6. Tests to ≥90% coverage