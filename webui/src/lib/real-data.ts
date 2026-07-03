/**
 * Maps real `/analyze_pipeline` responses (run against the canned
 * `DEMO_TRANSCRIPTS`) into the same `CallRow` shape the UI already renders
 * for mock data (`src/lib/mock-data.ts`). This is the "riktig data"
 * drop-in replacement referenced in docs/WEBUI_MODERNIZATION_PLAN.md §6:
 * the conversations are still synthetic/demo, but every number shown
 * (sentiment, QA score, risk, alerts) is computed by the real backend
 * pipeline instead of being hardcoded.
 *
 * Mirrors `get_overall_sentiment()` and
 * `app/archive/nicegui_dashboard/services/demo_provider.py::reports_to_table_rows()`.
 */

import type { PipelineReport } from "@/lib/api/client";
import type { DemoTranscript } from "@/lib/demo-transcripts";
import type { CallRow, RiskLevel, SentimentLabel } from "@/lib/mock-data";

const SENTIMENT_SCORE_MAP: Record<string, number> = {
  positiv: 0.8,
  positive: 0.8,
  neutral: 0.0,
  negativ: -0.7,
  negative: -0.7,
};

const SENTIMENT_LABEL_MAP: Record<string, SentimentLabel> = {
  positiv: "positive",
  positive: "positive",
  neutral: "neutral",
  negativ: "negative",
  negative: "negative",
};

export interface OverallSentiment {
  label: SentimentLabel;
  /** Normalized 0..1 score (0 = fully negative, 1 = fully positive) for display. */
  score: number;
}

/** Majority-vote sentiment across a call's segments, normalized to 0..1. */
export function getOverallSentiment(report: PipelineReport): OverallSentiment {
  const results = report.sentiment_results ?? [];
  const labels = results
    .map((s) => String(s?.label ?? "neutral").toLowerCase())
    .filter((label) => label.length > 0);

  if (labels.length === 0) {
    return { label: "neutral", score: 0.5 };
  }

  const counts = new Map<string, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  const majority = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];

  const avgRaw = labels.reduce((sum, l) => sum + (SENTIMENT_SCORE_MAP[l] ?? 0), 0) / labels.length;
  // Raw average is roughly -0.7..0.8; normalize to 0..1 for KPI/chart display.
  const normalized = Math.min(1, Math.max(0, (avgRaw + 1) / 2));

  return { label: SENTIMENT_LABEL_MAP[majority] ?? "neutral", score: normalized };
}

function riskLevelFromQa(report: PipelineReport): RiskLevel {
  const qa = report.results?.qa as Record<string, unknown> | undefined;
  const risks = report.risks as Record<string, unknown> | undefined;
  const level = (qa?.risk_level as string | undefined) ?? (risks?.risk_level as string | undefined);
  if (level === "low" || level === "medium" || level === "high" || level === "critical") {
    return level;
  }
  return "medium";
}

export interface RealCall {
  transcript: DemoTranscript;
  report: PipelineReport;
}

// ---------------------------------------------------------------------------
// Alert extraction (mirrors src/llm/schemas.py::Alert and
// app/services/data_services.py::collect_all_alerts)
// ---------------------------------------------------------------------------

export type AlertSeverity = "critical" | "high" | "medium" | "low" | "info";

export interface EvidenceSpan {
  text?: string;
  speaker?: string;
  start?: number;
  end?: number;
  turn_index?: number;
  [key: string]: unknown;
}

export interface AlertItem {
  rule_id: string;
  severity: AlertSeverity;
  message: string;
  evidence_spans?: EvidenceSpan[];
  recommended_actions?: string[];
  triggered_values?: Record<string, unknown>;
  source?: string;
  /** Added by collectAllAlerts for display context. */
  callId?: string;
  callTitle?: string;
  agent?: string;
}

const SEVERITY_ORDER: Record<AlertSeverity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
  info: 4,
};

function normalizeSeverity(s: unknown): AlertSeverity {
  const v = String(s ?? "info").toLowerCase();
  if (v === "critical" || v === "high" || v === "medium" || v === "low") return v;
  return "info";
}

/** Extract structured alerts from a single pipeline report. */
export function extractAlerts(report: PipelineReport): AlertItem[] {
  const raw = report.results?.alerts;
  if (!Array.isArray(raw)) return [];
  return raw.map((a) => ({
    rule_id: String(a?.rule_id ?? "unknown"),
    severity: normalizeSeverity(a?.severity),
    message: String(a?.message ?? ""),
    evidence_spans: Array.isArray(a?.evidence_spans) ? a.evidence_spans : [],
    recommended_actions: Array.isArray(a?.recommended_actions) ? a.recommended_actions : [],
    triggered_values: (a?.triggered_values as Record<string, unknown>) ?? {},
    source: a?.source ? String(a.source) : undefined,
  }));
}

/** Flatten alerts across all reports, with call context attached. */
export function collectAllAlerts(calls: RealCall[]): AlertItem[] {
  const all: AlertItem[] = [];
  for (const { transcript, report } of calls) {
    for (const alert of extractAlerts(report)) {
      all.push({
        ...alert,
        callId: transcript.id,
        callTitle: transcript.title,
        agent: transcript.meta.agent,
      });
    }
  }
  return all.sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]);
}

// ---------------------------------------------------------------------------
// LLM Judge extraction (mirrors src/llm/schemas.py::LLMJudgeResult)
// ---------------------------------------------------------------------------

export interface LlmJudgeVerdict {
  segment_index: number;
  original_sentiment: string;
  original_confidence: number;
  judge_label: string;
  judge_confidence: number;
  reasoning: string;
  model: string;
  cost_usd: number;
  latency_ms: number;
}

export interface LlmJudgeResult {
  verdicts: LlmJudgeVerdict[];
  triggered_segments: number;
  skipped_segments: number;
  total_cost_usd: number;
  budget_exceeded: boolean;
  fallback_used: boolean;
}

/** Extract LLM judge result from a pipeline report. Returns null if not run. */
export function extractLlmJudge(report: PipelineReport): LlmJudgeResult | null {
  const raw = report.results?.llm_judge ?? report.llm_judge ?? (report.llm as Record<string, unknown>)?.judge_verdicts;
  if (!raw || typeof raw !== "object") return null;

  const verdictsRaw = (raw as Record<string, unknown>)?.verdicts ?? (raw as Record<string, unknown>)?.results;
  if (!Array.isArray(verdictsRaw)) return null;

  const verdicts: LlmJudgeVerdict[] = verdictsRaw.map((v) => ({
    segment_index: Number(v?.segment_index ?? 0),
    original_sentiment: String(v?.original_sentiment ?? v?.original_label ?? "neutral"),
    original_confidence: Number(v?.original_confidence ?? v?.original_score ?? 0),
    judge_label: String(v?.judge_label ?? v?.label ?? "neutral"),
    judge_confidence: Number(v?.judge_confidence ?? v?.score ?? 0),
    reasoning: String(v?.reasoning ?? v?.explanation ?? ""),
    model: String(v?.model ?? ""),
    cost_usd: Number(v?.cost_usd ?? 0),
    latency_ms: Number(v?.latency_ms ?? 0),
  }));

  return {
    verdicts,
    triggered_segments: Number((raw as Record<string, unknown>)?.triggered_segments ?? verdicts.length),
    skipped_segments: Number((raw as Record<string, unknown>)?.skipped_segments ?? 0),
    total_cost_usd: Number((raw as Record<string, unknown>)?.total_cost_usd ?? 0),
    budget_exceeded: Boolean((raw as Record<string, unknown>)?.budget_exceeded ?? false),
    fallback_used: Boolean((raw as Record<string, unknown>)?.fallback_used ?? false),
  };
}

// ---------------------------------------------------------------------------
// Emotion timeline extraction (for /calls/[id] detail page)
// ---------------------------------------------------------------------------

export interface EmotionPoint {
  t: number;
  score: number;
}

/** Build an emotion timeline from sentiment_results + segment timestamps. */
export function extractEmotionTimeline(
  report: PipelineReport,
  segments: { start?: number; end?: number }[] = [],
): EmotionPoint[] {
  const sentiments = report.sentiment_results ?? [];
  const points: EmotionPoint[] = [];
  for (let i = 0; i < sentiments.length; i++) {
    const s = sentiments[i];
    const seg = segments[i] ?? {};
    const label = String(s?.label ?? "neutral").toLowerCase();
    const rawScore = SENTIMENT_SCORE_MAP[label] ?? 0;
    const normalized = Math.min(1, Math.max(0, (rawScore + 1) / 2));
    points.push({ t: Number(seg.start ?? i * 10), score: normalized });
  }
  return points;
}

// ---------------------------------------------------------------------------
// QA scorecard extraction (for /calls/[id] detail page)
// ---------------------------------------------------------------------------

export interface QaCriterion {
  criterion: string;
  passed: boolean;
  score: number;
  evidence: string;
}

export interface CallQa {
  score: number;
  passed: boolean;
  riskLevel: RiskLevel;
  complianceFlags: string[];
  criteria: QaCriterion[];
}

/** Extract a QA scorecard from a pipeline report. */
export function extractQa(report: PipelineReport): CallQa | null {
  const qa = report.results?.qa as Record<string, unknown> | undefined;
  if (!qa) return null;

  const criteriaRaw = Array.isArray(qa.criteria_results) ? qa.criteria_results : [];
  const criteria: QaCriterion[] = criteriaRaw.map((c) => ({
    criterion: String(c?.description ?? c?.id ?? ""),
    passed: Boolean(c?.passed ?? false),
    score: Number(c?.score ?? 0) * 100,
    evidence: Array.isArray(c?.evidence) ? c.evidence.join("; ") : String(c?.evidence ?? ""),
  }));

  const complianceFlags = Array.isArray(qa.compliance_flags) ? qa.compliance_flags.map(String) : [];

  return {
    score: Number(qa.overall_qa_score ?? 0),
    passed: Boolean(qa.passed ?? false),
    riskLevel: riskLevelFromQa(report),
    complianceFlags,
    criteria,
  };
}

// ---------------------------------------------------------------------------
// Call detail extraction (combines all above for /calls/[id])
// ---------------------------------------------------------------------------

export interface RealCallDetail {
  callId: string;
  transcript: { speaker: string; text: string; start: number }[];
  qa: CallQa | null;
  alerts: AlertItem[];
  llmJudge: LlmJudgeResult | null;
  emotionTimeline: EmotionPoint[];
  evidenceQuotes: string[];
}

/** Build a full call detail object from a RealCall (transcript + report). */
export function buildCallDetail({ transcript, report }: RealCall): RealCallDetail {
  const qa = extractQa(report);
  const alerts = extractAlerts(report);
  const llmJudge = extractLlmJudge(report);
  const emotionTimeline = extractEmotionTimeline(report, transcript.segments);

  // Evidence quotes from QA criteria + alert evidence
  const evidenceQuotes: string[] = [];
  if (qa) {
    for (const c of qa.criteria) {
      if (c.evidence && c.evidence.length > 0) evidenceQuotes.push(c.evidence);
    }
  }
  for (const a of alerts) {
    for (const e of a.evidence_spans ?? []) {
      if (e.text) evidenceQuotes.push(String(e.text));
    }
  }

  return {
    callId: transcript.id,
    transcript: transcript.segments.map((s) => ({
      speaker: s.speaker,
      text: s.text,
      start: s.start,
    })),
    qa,
    alerts,
    llmJudge,
    emotionTimeline,
    evidenceQuotes: evidenceQuotes.slice(0, 10),
  };
}

/** Map one real pipeline report (+ its source transcript metadata) to a CallRow. */
export function reportToCallRow({ transcript, report }: RealCall): CallRow {
  const overall = getOverallSentiment(report);
  const qa = report.results?.qa as { overall_qa_score?: number | null; passed?: boolean } | undefined;
  const alerts = report.results?.alerts;
  const alertCount = Array.isArray(alerts) ? alerts.length : 0;

  return {
    callId: transcript.id,
    title: transcript.title,
    agent: transcript.meta.agent,
    category: transcript.meta.category,
    sentiment: overall.label,
    sentimentScore: overall.score,
    riskLevel: riskLevelFromQa(report),
    alertCount,
    qaPassed: qa?.passed ?? null,
    qaScore: qa?.overall_qa_score ?? null,
    durationS: transcript.meta.duration_s,
    timestamp: new Date().toISOString(),
  };
}

export function reportsToCallRows(calls: RealCall[]): CallRow[] {
  return calls.map(reportToCallRow);
}
