/**
 * Typed fetch client for the FastAPI backend.
 *
 * Talks to the same REST endpoints as the Python API without backend changes.
 * Path inventory is checked against generated OpenAPI types in `./paths.ts`
 * (`npm run generate:types`). Hand-written response shapes below remain the
 * runtime contract until callers migrate onto `schema.ts` components.
 */

import "./paths";

const DEFAULT_BASE_URL = "http://localhost:8000";

function truthyEnv(value: string | undefined): boolean {
  const flag = value?.trim().toLowerCase();
  return flag === "1" || flag === "true" || flag === "yes";
}

function falsyEnv(value: string | undefined): boolean {
  const flag = value?.trim().toLowerCase();
  return flag === "0" || flag === "false" || flag === "no";
}

/** When true, browser REST talks to FastAPI directly (legacy trusted-LAN). */
function isDirectApiEnabled(): boolean {
  return truthyEnv(process.env.NEXT_PUBLIC_USE_DIRECT_API);
}

/** BFF proxy is the default so the API key stays server-side. */
function isApiProxyEnabled(): boolean {
  if (isDirectApiEnabled()) return false;
  if (falsyEnv(process.env.NEXT_PUBLIC_USE_API_PROXY)) return false;
  return true;
}

function getBaseUrl(): string {
  if (isApiProxyEnabled()) {
    return "/api/backend";
  }
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_BASE_URL;
}

/** Direct FastAPI origin for WebSocket (BFF does not proxy WS). */
function getDirectApiBaseUrl(): string {
  const explicit =
    process.env.NEXT_PUBLIC_WS_API_BASE_URL?.trim() ||
    process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (isApiProxyEnabled() && !explicit) {
    throw new ApiError(
      "WebSocket kräver NEXT_PUBLIC_WS_API_BASE_URL eller NEXT_PUBLIC_API_BASE_URL när BFF-proxy är på",
    );
  }
  return (explicit || DEFAULT_BASE_URL).replace(/\/$/, "");
}

/** Browser-visible API key (legacy trusted-LAN). Unused when BFF proxy is enabled. */
function getApiKeyFromEnv(): string | undefined {
  if (isApiProxyEnabled()) {
    return undefined;
  }
  const key =
    process.env.NEXT_PUBLIC_API_KEY?.trim() ||
    process.env.NEXT_PUBLIC_SENTIMENT_API_KEY?.trim();
  return key || undefined;
}

/** Connection + auth probe result for the header badge. */
export type ApiConnectionStatus = {
  reachable: boolean;
  /** null = auth not required or not probed */
  authenticated: boolean | null;
  status: "ok" | "offline" | "unauthorized" | "auth_required" | "degraded";
  detail?: string;
};

/** Loose shape of a CallAnalysisReport dict returned by /analyze_pipeline. */
export interface PipelineReport {
  sentiment_results?: { label?: string; score?: number }[];
  results?: {
    qa?: { overall_qa_score?: number | null; [key: string]: unknown };
    alerts?: Record<string, unknown>[];
    llm_judge?: Record<string, unknown>;
    [key: string]: unknown;
  };
  llm?: {
    actionable_summary?: { problem?: string; [key: string]: unknown };
    judge_verdicts?: unknown;
    [key: string]: unknown;
  };
  llm_judge?: Record<string, unknown>;
  risks?: Record<string, unknown>;
  insights?: Record<string, unknown>;
  /** Typed view of `results` (Fas 5). Null when no analyzers ran. */
  analyzer_results?: AnalyzerResults | null;
  /** Graceful-degradation reasons from the API. */
  degraded?: string[];
  /** 'full' | 'degraded' */
  mode?: string;
  [key: string]: unknown;
}

export interface ModelCompareResult {
  model: string;
  processing_time_s: number;
  llm_cost_usd?: number | null;
  qa_score?: number | null;
  sentiment_label?: string | null;
  llm_trajectory?: string | null;
  response: PipelineReport;
}

export interface PipelineCompareResponse {
  models: string[];
  results: Record<string, ModelCompareResult>;
  total_cost_usd?: number | null;
  total_processing_time_s: number;
  budget_usd?: number | null;
  budget_exceeded: boolean;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Typed analyzer output interfaces (Fas 5 — mirrors src/api/schemas.py)
// ---------------------------------------------------------------------------

export interface EmotionSegmentResult {
  primary: string;
  scores: Record<string, number>;
  speaker?: string | null;
}

export interface AspectItem {
  aspect: string;
  sentiment: string;
  score: number;
  evidence?: string | null;
  evidence_spans?: Array<{
    text: string;
    speaker_role?: string | null;
    turn_index?: number | null;
    segment_id?: number | null;
    start?: number | null;
    end?: number | null;
  }> | null;
  start?: number | null;
  end?: number | null;
  speaker?: string | null;
  source?: string | null;
  related_to?: string[] | null;
}

export interface DerivedCallSentiment {
  label: string;
  score: number;
  aspect_count: number;
  by_aspect: Record<string, number>;
  source: string;
}

export interface TrajectoryResult {
  customer_sentiment_slope: number;
  escalation_events: number;
  escalation_event_details: Array<Record<string, unknown>>;
  peak_frustration_turn: number | null;
  sentiment_trend: number[];
}

export interface RootCauseItem {
  cause: string;
  count: number;
  recommendation?: string | null;
}

export interface RootCauseResult {
  root_causes: RootCauseItem[];
  top_root_cause?: string | null;
  evidence_examples: Array<Record<string, unknown>>;
  overall_risk: "low" | "medium" | "high";
  message?: string | null;
}

export interface CoachingInsightItem {
  rule_id: string;
  priority: "high" | "medium" | "low";
  recommendation: string;
  evidence?: unknown;
}

export interface CoachingResult {
  coaching_insights: CoachingInsightItem[];
  top_recommendation?: string | null;
  insight_count: number;
}

export interface CustomerEffortSegment {
  speaker?: string | null;
  start: number;
  end: number;
  effort_score: number;
}

export interface CustomerEffortResult {
  overall_ces: number;
  scale: string;
  per_segment: CustomerEffortSegment[];
  coaching_tips: string[];
}

export interface ActiveListeningResult {
  listening_score: number;
  backchannel_count: number;
  speaker_balance: Record<string, number>;
  events: Array<Record<string, unknown>>;
  tips: string[];
}

export interface EmpathySegment {
  speaker?: string | null;
  start?: number | null;
  empathy_score: number;
  evidence: string[];
}

export interface EmpathyResult {
  overall_empathy: number;
  scale: string;
  per_segment: EmpathySegment[];
  coaching_tips: string[];
}

export interface ResolutionProbabilityResult {
  resolution_probability: number;
  confidence: number;
  recommended_action: string;
  factors: Record<string, unknown>;
}

export interface JourneyStage {
  stage: string;
  start: number;
  speaker?: string | null;
  text_snippet: string;
  intent?: string | null;
  sentiment?: string | null;
}

export interface MultiTurnJourneyResult {
  journey_stages: JourneyStage[];
  resolved: boolean;
  unresolved_count: number;
  key_turning_points: Array<Record<string, unknown>>;
  recommendation?: string | null;
  message?: string | null;
}

export interface UpsellOpportunity {
  speaker?: string | null;
  start: number;
  end: number;
  confidence: number;
  signals: string[];
  suggested_action: string;
  evidence: string;
}

export interface UpsellResult {
  opportunities: UpsellOpportunity[];
  count: number;
  recommendation?: string | null;
}

export interface DialectSensitivityResult {
  dialect_risk_level: "low" | "medium" | "high";
  flagged_segments: Array<Record<string, unknown>>;
  total_dialect_hits: number;
  slang_count: number;
  recommendation?: string | null;
}

export interface ComplianceRiskResult {
  overall_risk_level: "low" | "medium" | "high";
  flagged_segments: Array<Record<string, unknown>>;
  recommendation?: string | null;
}

export interface RoleClassifierResult {
  roles: Record<string, string>;
  talk_ratios: Record<string, number>;
  question_density: Record<string, number>;
  lexical_formality: number;
  intervention_count: number;
  sentiment_variance: number;
  num_agent_turns: number;
  num_customer_turns: number;
}

export interface PredictiveResult {
  churn_risk: number;
  escalation_risk: number;
  satisfaction_score: number;
  risk_factors: string[];
  risk_level: "low" | "medium" | "high" | "critical";
  recommended_action?: string | null;
}

/** Typed view of `PipelineResponse.results` (Fas 5). */
export interface AnalyzerResults {
  emotion?: EmotionSegmentResult[] | null;
  aspect?: AspectItem[] | null;
  trajectory?: TrajectoryResult | null;
  root_cause?: RootCauseResult | null;
  actionable_coaching?: CoachingResult | null;
  customer_effort?: CustomerEffortResult | null;
  active_listening?: ActiveListeningResult | null;
  empathy?: EmpathyResult | null;
  resolution_probability?: ResolutionProbabilityResult | null;
  multi_turn_journey?: MultiTurnJourneyResult | null;
  upsell_opportunity?: UpsellResult | null;
  dialect_sensitivity?: DialectSensitivityResult | null;
  compliance_risk?: ComplianceRiskResult | null;
  role?: RoleClassifierResult | null;
  predictive?: PredictiveResult | null;
  agent_performance?: Record<string, unknown> | null;
  qa?: Record<string, unknown> | null;
  agent_assessment?: Record<string, unknown> | null;
  agent_assessment_local?: Record<string, unknown> | null;
  customer_metrics?: Record<string, unknown> | null;
  pii_redaction?: Record<string, unknown> | null;
  alerts?: Array<Record<string, unknown>> | null;
  llm_judge?: Record<string, unknown> | null;
  override_provenance?: OverrideProvenanceEntry[] | null;
  deep_path_ccp?: DeepPathCCP | null;
  degradation?: DegradationInfo | null;
  partial?: PartialAnalysisMeta | null;
  analyzer_routing?: AnalyzerRouting | null;
}

export interface CCPCheck {
  name: string;
  passed: boolean;
  detail: string;
  corrective_action?: string | null;
}

export interface DeepPathCCP {
  passed: boolean;
  checks: CCPCheck[];
  failed?: string[];
}

export interface DegradationInfo {
  mode: string;
  deep_path_active: boolean;
  llm_used: boolean;
}

export interface AnalyzerRouting {
  profile_prior?: string[];
  pre_selected?: string[];
  runtime_selected?: string[];
  extras_run?: string[];
  segment_count?: number;
  applied?: boolean;
}

export interface OverrideProvenanceEntry {
  field: string;
  source?: string | null;
  evidence_spans?: Array<{ text: string; speaker_role?: string | null }>;
}

export interface PartialAnalysisMeta {
  incremental?: boolean;
  reconciled?: boolean;
  [key: string]: unknown;
}

/** Response shape of POST /agent_performance/{agent_id} (Fas 4). */
export interface AgentPerformanceResponse {
  agent_id: string;
  metrics: {
    call_count?: number;
    averages?: Record<string, number>;
    trend_empathy?: string;
    /** Count of compliance flags across the agent's calls (aggregate endpoint does not return the flag text itself). */
    total_compliance_flags?: number;
    avg_flags_per_call?: number;
    [key: string]: unknown;
  };
  cached?: boolean;
  timestamp: string;
}

/** Response shape of POST /insights/hot_topics (Fas 4). */
export interface HotTopicItem {
  topic: string;
  volume: number;
  avg_sentiment: number;
  trend: "up" | "down" | "stable";
  evidence_spans?: unknown[];
  sample_quotes?: string[];
  llm_summary?: string | null;
}

export interface HotTopicsResponse {
  hot_topics: HotTopicItem[];
  meta: Record<string, unknown>;
  timestamp: string;
}

/** Response shape of GET /alerting/status (webhook + circuit breaker health). */
export interface AlertingStatusResponse {
  ok?: boolean;
  webhook?: {
    circuit_breaker_open?: boolean;
    consecutive_failures?: number;
    circuit_breaker_threshold?: number;
    [key: string]: unknown;
  };
  note?: string;
  [key: string]: unknown;
}

/** Response shape of POST /alerts (Fas 4.4.2 + 4.5.2). */
export interface AlertsResponse {
  alerts: Record<string, unknown>[];
  timestamp: string;
}

/** Response shape of POST /search/semantic. */
export interface SemanticSearchResponse {
  query: string;
  hits: Array<{
    text?: string;
    score?: number;
    call_index?: number;
    segment_index?: number;
    speaker?: string;
    [key: string]: unknown;
  }>;
  meta: Record<string, unknown>;
  timestamp: string;
}

/** Response shape of POST /qa/score. */
export interface QAScoreResponse {
  qa: {
    overall_qa_score?: number | null;
    passed?: boolean;
    criteria_results?: unknown[];
    compliance_flags?: unknown[];
    [key: string]: unknown;
  };
  timestamp: string;
}

/** Response shape of GET /transcription/jobs. */
export interface TranscriptionJobStatus {
  job_id: string;
  kind: string;
  status: string;
  created_at: string;
  [key: string]: unknown;
}

export interface TranscriptionJobListResponse {
  jobs: TranscriptionJobStatus[];
  [key: string]: unknown;
}

/** Edge AI: single segment result from offline analysis. */
export interface EdgeSegmentResult {
  text: string;
  sentiment_label: string | null;
  sentiment_score: number | null;
  intent: string | null;
}

/** Edge AI: full offline analysis result (POST /edge/analyze-*). */
export interface EdgeAnalysisResult {
  profile: string;
  offline: boolean;
  llm_used: boolean;
  segments: EdgeSegmentResult[];
  summary: string;
  limitations: string[];
}

/** Response shape of POST /transcribe. */
export interface TranscribeResponse {
  transcript: Record<string, unknown>;
  timestamp: string;
}

/** Request shape of POST /transcribe (subset of backend AsrParamsMixin). */
export interface TranscribeRequest {
  audio_path: string;
  backend?: string;
  model?: string;
  device?: string;
  language?: string;
  word_timestamps?: boolean;
  preprocess?: boolean;
  preprocess_mode?: string;
  vad?: boolean;
  diarize?: boolean;
  num_speakers?: number | null;
}

/** Response shape of POST /batch_transcribe. */
export interface BatchTranscribeResponse {
  items: Array<{
    file: string;
    transcript?: Record<string, unknown>;
    error?: string;
  }>;
  ok: number;
  failed: number;
  total: number;
  timestamp: string;
}

/** Response shape of POST /upload. */
export interface UploadResponse {
  audio_path: string;
  filename: string;
  size_bytes: number;
  timestamp: string;
}

/** Request shape of POST /batch_transcribe. */
export interface BatchTranscribeRequest {
  audio_paths?: string[];
  directory?: string | null;
  glob?: string | null;
  recursive?: boolean;
  limit?: number | null;
  workers?: number;
  worker_timeout?: number;
  backend?: string;
  model?: string;
  device?: string;
  language?: string;
  word_timestamps?: boolean;
  vad?: boolean;
  diarize?: boolean;
  num_speakers?: number | null;
}

export class ApiError extends Error {
  status?: number;
  detail?: unknown;

  constructor(message: string, status?: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export interface ApiClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private wsTicket: string | null = null;
  private wsTicketExpiry: number = 0;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? getBaseUrl()).replace(/\/$/, "");
    this.apiKey = options.apiKey ?? getApiKeyFromEnv();
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  /** True when a browser API key is configured, or when BFF proxy handles auth. */
  get hasApiKey(): boolean {
    return Boolean(this.apiKey) || isApiProxyEnabled();
  }

  get usesProxy(): boolean {
    return isApiProxyEnabled();
  }

  private headers(opts: { json?: boolean } = {}): HeadersInit {
    const headers: Record<string, string> = {};
    if (opts.json !== false) {
      headers["Content-Type"] = "application/json";
    }
    if (this.apiKey) headers["X-API-Key"] = this.apiKey;
    return headers;
  }

  private async request<T>(path: string, init: RequestInit = {}, timeoutMs?: number): Promise<T> {
    const effectiveTimeout = timeoutMs ?? this.timeoutMs;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), effectiveTimeout);
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: { ...this.headers(), ...(init.headers ?? {}) },
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new ApiError(`Timeout mot ${path} (${effectiveTimeout}ms)`);
      }
      throw new ApiError(`Kan inte ansluta till backend (${this.baseUrl}): ${String(err)}`);
    } finally {
      clearTimeout(timeout);
    }

    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = await response.text();
      }
      throw new ApiError(`API-fel ${response.status} på ${path}`, response.status, detail);
    }
    return (await response.json()) as T;
  }

  /** Fetch a WebSocket authentication ticket (for browsers that cannot send custom headers). */
  private async fetchWsTicket(): Promise<string> {
    const now = Date.now();
    // Reuse existing ticket if it's still valid (with 30s buffer)
    if (this.wsTicket && this.wsTicketExpiry > now + 30_000) {
      return this.wsTicket;
    }

    try {
      const response = await this.get<{ ticket: string; expires_in: number }>("/ws/transcription/ticket");
      this.wsTicket = response.ticket;
      this.wsTicketExpiry = now + response.expires_in * 1000;
      return this.wsTicket;
    } catch (err) {
      this.wsTicket = null;
      this.wsTicketExpiry = 0;
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        throw err;
      }
      return "";
    }
  }

  get<T>(path: string, params?: Record<string, string | number | boolean | undefined>) {
    const search = params
      ? "?" +
        new URLSearchParams(
          Object.entries(params)
            .filter(([, v]) => v !== undefined)
            .map(([k, v]) => [k, String(v)]),
        ).toString()
      : "";
    return this.request<T>(`${path}${search}`, { method: "GET" });
  }

  post<T>(path: string, body: unknown, timeoutMs?: number) {
    return this.request<T>(path, { method: "POST", body: JSON.stringify(body) }, timeoutMs);
  }

  async health(): Promise<boolean> {
    const status = await this.connectionStatus();
    return status.status === "ok";
  }

  /**
   * Probe /health (public) then a protected ticket endpoint to surface 401s.
   * When auth is disabled, ticket returns 200 and status is "ok".
   */
  async connectionStatus(): Promise<ApiConnectionStatus> {
    try {
      const healthRes = await fetch(`${this.baseUrl}/ready`, {
        signal: AbortSignal.timeout(10_000),
      });
      if (healthRes.status >= 500) {
        return {
          reachable: true,
          authenticated: null,
          status: "degraded",
          detail: `Backend /ready ${healthRes.status}`,
        };
      }
      if (!healthRes.ok && healthRes.status !== 503) {
        const fallback = await fetch(`${this.baseUrl}/health`, {
          signal: AbortSignal.timeout(10_000),
        });
        if (!fallback.ok) {
          return { reachable: false, authenticated: null, status: "offline" };
        }
      }
    } catch {
      return { reachable: false, authenticated: null, status: "offline" };
    }

    try {
      await this.get<{ ticket: string }>("/ws/transcription/ticket");
      return {
        reachable: true,
        authenticated: this.hasApiKey ? true : null,
        status: "ok",
      };
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        if (!this.hasApiKey) {
          return {
            reachable: true,
            authenticated: false,
            status: "auth_required",
            detail: isApiProxyEnabled()
              ? "Backend kräver auth — sätt SENTIMENT_API_KEY på Next.js-servern (BFF)"
              : "Backend kräver X-API-Key — sätt SENTIMENT_API_KEY på BFF eller NEXT_PUBLIC_USE_DIRECT_API=1 + NEXT_PUBLIC_API_KEY",
          };
        }
        return {
          reachable: true,
          authenticated: false,
          status: "unauthorized",
          detail: "API-nyckel avvisad (401)",
        };
      }
      if (err instanceof ApiError && err.status != null && err.status >= 500) {
        return {
          reachable: true,
          authenticated: null,
          status: "degraded",
          detail: err.message,
        };
      }
      return {
        reachable: true,
        authenticated: null,
        status: "ok",
        detail: err instanceof Error ? err.message : String(err),
      };
    }
  }

  /** Cost-aware analysis perspectives with recommended paid models. */
  getAnalysisProfiles(options: { top_k?: number; refresh?: boolean } = {}) {
    const q = new URLSearchParams();
    if (options.top_k != null) q.set("top_k", String(options.top_k));
    if (options.refresh != null) q.set("refresh", String(options.refresh));
    const qs = q.toString();
    return this.get<import("@/lib/analysis-profiles").AnalysisProfilesResponse>(
      `/llm/analysis-profiles${qs ? `?${qs}` : ""}`,
    );
  }

  getAnalysisProfileDetail(perspectiveId: string, top_k = 5) {
    return this.get<Record<string, unknown>>(
      `/llm/analysis-profiles/${encodeURIComponent(perspectiveId)}?top_k=${top_k}`,
    );
  }

  analyzePipeline<T = PipelineReport>(
    segments: unknown[],
    options: Record<string, unknown> = {},
  ) {
    // The full CallAnalysisPipeline (sentiment + QA + insights + Fas 4
    // analyzers) can take 30–60s per call on CPU. Use a generous timeout so
    // the demo transcripts don't get aborted mid-analysis.
    return this.post<T>("/analyze_pipeline", { segments, profile: "callcenter", ...options }, 180_000);
  }

  analyzePipelinePartial<T = PipelineReport>(
    segments: unknown[],
    options: {
      previous_results?: Record<string, unknown> | null;
      reconcile?: boolean;
      profile?: string;
      deep_analysis?: boolean;
      use_mistral_llm?: boolean;
      [key: string]: unknown;
    } = {},
  ) {
    const { previous_results, reconcile = false, profile = "callcenter", ...rest } = options;
    return this.post<T>(
      "/analyze_pipeline/partial",
      {
        segments,
        profile,
        previous_results: previous_results ?? null,
        reconcile,
        ...rest,
      },
      180_000,
    );
  }

  /** Compare up to 3 LLM models on the same segments (v0.5 model A/B). */
  comparePipeline(
    segments: unknown[],
    models: string[],
    options: Record<string, unknown> = {},
  ) {
    return this.post<PipelineCompareResponse>(
      "/analyze_pipeline/compare",
      {
        segments,
        models,
        profile: "callcenter",
        deep_analysis: true,
        ...options,
      },
      540_000,
    );
  }

  /** Aggregate agent metrics for one agent, computed over the given calls (Fas 4). */
  getAgentPerformance<T = AgentPerformanceResponse>(
    agentId: string,
    segmentsList: unknown[][],
    options: Record<string, unknown> = {},
  ) {
    return this.post<T>(
      `/agent_performance/${encodeURIComponent(agentId)}`,
      {
        segments_list: segmentsList,
        agent_id: agentId,
        profile: "callcenter",
        window: "7d",
        ...options,
      },
      180_000,
    );
  }

  /** Hot topics aggregated across the given calls (Fas 4). */
  getHotTopics<T = HotTopicsResponse>(segmentsList: unknown[][], options: Record<string, unknown> = {}) {
    return this.post<T>(
      "/insights/hot_topics",
      {
        segments_list: segmentsList,
        profile: "callcenter",
        window: "7d",
        ...options,
      },
      180_000,
    );
  }

  semanticSearch<T = SemanticSearchResponse>(
    query: string,
    segmentsList: unknown[][],
    options: { top_k?: number; filters?: Record<string, unknown> } & Record<string, unknown> = {},
  ) {
    const { top_k = 5, filters, ...rest } = options;
    return this.post<T>(
      "/search/semantic",
      {
        query,
        segments_list: segmentsList,
        top_k,
        filters: filters ?? null,
        profile: "callcenter",
        ...rest,
      },
      180_000,
    );
  }

  getQaScore<T = QAScoreResponse>(segments: unknown[], options: Record<string, unknown> = {}) {
    return this.post<T>(
      "/qa/score",
      {
        segments,
        profile: "callcenter",
        ...options,
      },
      180_000,
    );
  }

  getAlertingStatus<T = AlertingStatusResponse>() {
    return this.get<T>("/alerting/status");
  }

  /** Manually reset the webhook circuit breaker (POST /alerting/reset-circuit-breaker). */
  resetCircuitBreaker<T = AlertingStatusResponse>() {
    return this.post<T>("/alerting/reset-circuit-breaker", {});
  }

  /** Get alerts from per-call results or aggregate trends (POST /alerts, Fas 4). */
  getAlerts<T = AlertsResponse>(
    segmentsList: unknown[][] | null = null,
    aggregate: Record<string, unknown> | null = null,
    options: Record<string, unknown> = {},
  ) {
    return this.post<T>(
      "/alerts",
      {
        segments_list: segmentsList,
        aggregate,
        profile: "callcenter",
        ...options,
      },
      180_000,
    );
  }

  getProcessEvents<T = unknown>(params: { limit?: number; job_id?: string; component?: string; level?: string } = {}) {
    return this.get<T>("/status/processes", { limit: params.limit ?? 100, ...params });
  }

  getJobStatus<T = unknown>(jobId: string) {
    return this.get<T>(`/status/jobs/${jobId}`);
  }

  listTranscriptionJobs<T = TranscriptionJobListResponse>(limit = 20) {
    return this.get<T>("/transcription/jobs", { limit });
  }

  getTranscriptionJob<T = TranscriptionJobStatus>(jobId: string) {
    return this.get<T>(`/transcription/jobs/${jobId}`);
  }

  cancelTranscriptionJob<T = { job_id: string; cancelled: boolean; status?: string }>(jobId: string) {
    return this.post<T>(`/transcription/jobs/${jobId}/cancel`, {});
  }

  /** Transcribe a single audio file (POST /transcribe). */
  transcribe<T = TranscribeResponse>(req: TranscribeRequest) {
    return this.post<T>("/transcribe", req, 600_000); // 10 minute timeout for large files
  }

  /** Transcribe multiple audio files (POST /batch_transcribe). */
  batchTranscribe<T = BatchTranscribeResponse>(req: BatchTranscribeRequest) {
    return this.post<T>("/batch_transcribe", req, 600_000); // 10 minute timeout for batches
  }

  /** Upload an audio file (POST /upload). */
  async upload<T = UploadResponse>(file: File, options: { timeoutMs?: number } = {}): Promise<T> {
    const formData = new FormData();
    formData.append("file", file);
    const timeoutMs = options.timeoutMs ?? 300_000; // 5 minute default for audio uploads up to 200MB
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/upload`, {
        method: "POST",
        headers: this.headers({ json: false }),
        body: formData,
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new ApiError(`Timeout mot /upload (${timeoutMs}ms)`);
      }
      throw new ApiError(`Kan inte ansluta till backend (${this.baseUrl}): ${String(err)}`);
    } finally {
      clearTimeout(timeout);
    }

    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = await response.text();
      }
      throw new ApiError(`Upload failed: ${response.status}`, response.status, detail);
    }

    return (await response.json()) as T;
  }

  /** ws:// or wss:// URL for the live transcription event stream. */
  async wsUrl(path = "/ws/transcription"): Promise<string> {
    // WebSocket cannot go through the REST BFF — always use the direct API origin.
    const origin = isApiProxyEnabled() ? getDirectApiBaseUrl() : this.baseUrl;
    const url = new URL(origin.startsWith("http") ? origin : DEFAULT_BASE_URL);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = path;

    // Fetch and append WebSocket ticket for auth (browsers cannot send custom headers)
    const ticket = await this.fetchWsTicket();
    if (ticket) {
      url.searchParams.set("token", ticket);
    } else {
      url.search = "";
    }
    return url.toString();
  }

  /** List server-persisted analyzed calls. */
  listCalls<T = { calls: unknown[]; count: number }>(limit = 50) {
    return this.get<T>("/calls", { limit });
  }

  /** Persist an analyzed call on the server (localStorage remains a cache). */
  saveCall<T = Record<string, unknown>>(
    id: string,
    body: {
      transcript?: Record<string, unknown>;
      report?: Record<string, unknown>;
      meta?: Record<string, unknown>;
      created_at?: string;
    },
  ) {
    return this.post<T>("/calls", { id, ...body });
  }

  deleteCall<T = { id: string; deleted: boolean }>(id: string) {
    return this.request<T>(`/calls/${encodeURIComponent(id)}`, { method: "DELETE" });
  }

  /** Edge AI: analyze a single text offline (POST /edge/analyze-text). */
  edgeAnalyzeText<T = EdgeAnalysisResult>(text: string, profile = "callcenter") {
    return this.post<T>("/edge/analyze-text", { text, profile }, 60_000);
  }

  /** Edge AI: analyze pre-transcribed segments offline (POST /edge/analyze-segments). */
  edgeAnalyzeSegments<T = EdgeAnalysisResult>(
    segments: { text: string; speaker?: string }[],
    profile = "callcenter",
  ) {
    return this.post<T>("/edge/analyze-segments", { segments, profile }, 60_000);
  }
}

export const apiClient = new ApiClient({
  apiKey: getApiKeyFromEnv(),
});
