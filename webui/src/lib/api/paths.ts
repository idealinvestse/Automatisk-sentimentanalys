/**
 * Compile-time inventory: ApiClient paths must exist in generated OpenAPI types.
 *
 * Regenerate after backend schema changes:
 *   python scripts/export_openapi.py -o webui/openapi.json
 *   cd webui && npm run generate:types
 */
import type { paths } from "./schema";

/** Assert that a path key exists on the OpenAPI `paths` map. */
type ApiPath = keyof paths;

const _clientPaths = [
  "/health",
  "/ready",
  "/calls",
  "/calls/{call_id}",
  "/llm/analysis-profiles",
  "/llm/analysis-profiles/{perspective_id}",
  "/llm/providers",
  "/analyze_pipeline",
  "/analyze_pipeline/partial",
  "/analyze_pipeline/compare",
  "/agent_performance/{agent_id}",
  "/insights/hot_topics",
  "/search/semantic",
  "/qa/score",
  "/alerts",
  "/alerting/status",
  "/alerting/reset-circuit-breaker",
  "/status/processes",
  "/status/jobs/{job_id}",
  "/transcription/jobs",
  "/transcription/jobs/{job_id}",
  "/transcription/jobs/{job_id}/cancel",
  "/upload",
  "/transcribe",
  "/batch_transcribe",
  "/ws/transcription/ticket",
  "/edge/analyze-text",
  "/edge/analyze-segments",
] as const satisfies readonly ApiPath[];

void _clientPaths;

/** JSON body for successful POST /analyze_pipeline (OpenAPI). */
export type OpenApiPipelineResponse = NonNullable<
  paths["/analyze_pipeline"]["post"]["responses"][200]["content"]["application/json"]
>;

/** JSON body for POST /insights/hot_topics. */
export type OpenApiHotTopicsResponse = NonNullable<
  paths["/insights/hot_topics"]["post"]["responses"][200]["content"]["application/json"]
>;

/** JSON body for GET /ws/transcription/ticket. */
export type OpenApiWsTicketResponse = NonNullable<
  paths["/ws/transcription/ticket"]["get"]["responses"][200]["content"]["application/json"]
>;

export type { paths };
