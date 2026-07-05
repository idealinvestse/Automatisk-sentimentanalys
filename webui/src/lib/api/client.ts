/**
 * Typed fetch client for the FastAPI backend.
 *
 * Mirrors app/nicegui_dashboard/services/nicegui_api_client.py so the new
 * web UI talks to the exact same REST endpoints as the legacy NiceGUI
 * dashboard. No backend changes are required to use this client.
 */

const DEFAULT_BASE_URL = "http://localhost:8000";

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
  [key: string]: unknown;
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
  start?: number | null;
  end?: number | null;
  speaker?: string | null;
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

function getBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_BASE_URL;
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private wsTicket: string | null = null;
  private wsTicketExpiry: number = 0;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? getBaseUrl()).replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  private headers(): HeadersInit {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
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
    } catch {
      // If ticket fetch fails (e.g., no auth), fall back to no ticket
      // This allows the WS to work when auth is disabled
      this.wsTicket = null;
      this.wsTicketExpiry = 0;
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
    try {
      const response = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(10_000) });
      return response.ok;
    } catch {
      return false;
    }
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

  listTranscriptionJobs<T = unknown>(limit = 20) {
    return this.get<T>("/transcription/jobs", { limit });
  }

  getTranscriptionJob<T = unknown>(jobId: string) {
    return this.get<T>(`/transcription/jobs/${jobId}`);
  }

  cancelTranscriptionJob<T = unknown>(jobId: string) {
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

  /** ws:// or wss:// URL for the live transcription event stream. */
  async wsUrl(path = "/ws/transcription"): Promise<string> {
    const url = new URL(this.baseUrl);
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

export const apiClient = new ApiClient();
