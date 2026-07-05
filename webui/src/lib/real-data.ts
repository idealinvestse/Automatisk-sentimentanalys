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
// Fas 5: Typed analyzer extraction (uses analyzer_results when available,
// falls back to raw results dict for backward compatibility)
// ---------------------------------------------------------------------------

import type {
  EmotionSegmentResult,
  AspectItem,
  TrajectoryResult,
  RootCauseResult,
  CoachingResult,
  CustomerEffortResult,
  ActiveListeningResult,
  EmpathyResult,
  ResolutionProbabilityResult,
  MultiTurnJourneyResult,
  UpsellResult,
  DialectSensitivityResult,
  ComplianceRiskResult,
  RoleClassifierResult,
  PredictiveResult,
} from "@/lib/api/client";

/** Extract emotion labels per segment. Returns [] if analyzer didn't run. */
export function extractEmotion(report: PipelineReport): EmotionSegmentResult[] {
  const typed = report.analyzer_results?.emotion;
  if (typed) return typed;
  const raw = report.results?.emotion;
  if (!Array.isArray(raw)) return [];
  return raw as EmotionSegmentResult[];
}

/** Extract aspect-based sentiment items. Returns [] if analyzer didn't run. */
export function extractAspects(report: PipelineReport): AspectItem[] {
  const typed = report.analyzer_results?.aspect;
  if (typed) return typed;
  const raw = report.results?.aspect;
  if (!Array.isArray(raw)) return [];
  return raw as AspectItem[];
}

/** Extract trajectory result. Returns null if analyzer didn't run. */
export function extractTrajectory(report: PipelineReport): TrajectoryResult | null {
  const typed = report.analyzer_results?.trajectory;
  if (typed) return typed;
  const raw = report.results?.trajectory;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as TrajectoryResult;
}

/** Extract root cause result. Returns null if analyzer didn't run. */
export function extractRootCause(report: PipelineReport): RootCauseResult | null {
  const typed = report.analyzer_results?.root_cause;
  if (typed) return typed;
  const raw = report.results?.root_cause;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as RootCauseResult;
}

/** Extract actionable coaching result. Returns null if analyzer didn't run. */
export function extractCoaching(report: PipelineReport): CoachingResult | null {
  const typed = report.analyzer_results?.actionable_coaching;
  if (typed) return typed;
  const raw = report.results?.actionable_coaching;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as CoachingResult;
}

/** Extract customer effort score. Returns null if analyzer didn't run. */
export function extractCustomerEffort(report: PipelineReport): CustomerEffortResult | null {
  const typed = report.analyzer_results?.customer_effort;
  if (typed) return typed;
  const raw = report.results?.customer_effort;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as CustomerEffortResult;
}

/** Extract active listening result. Returns null if analyzer didn't run. */
export function extractActiveListening(report: PipelineReport): ActiveListeningResult | null {
  const typed = report.analyzer_results?.active_listening;
  if (typed) return typed;
  const raw = report.results?.active_listening;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as ActiveListeningResult;
}

/** Extract empathy result (per-segment + overall). Returns null if not run. */
export function extractEmpathy(report: PipelineReport): EmpathyResult | null {
  const typed = report.analyzer_results?.empathy;
  if (typed) return typed;
  const raw = report.results?.empathy;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as EmpathyResult;
}

/** Extract resolution probability. Returns null if analyzer didn't run. */
export function extractResolutionProbability(
  report: PipelineReport,
): ResolutionProbabilityResult | null {
  const typed = report.analyzer_results?.resolution_probability;
  if (typed) return typed;
  const raw = report.results?.resolution_probability;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as ResolutionProbabilityResult;
}

/** Extract multi-turn journey. Returns null if analyzer didn't run. */
export function extractJourney(report: PipelineReport): MultiTurnJourneyResult | null {
  const typed = report.analyzer_results?.multi_turn_journey;
  if (typed) return typed;
  const raw = report.results?.multi_turn_journey;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as MultiTurnJourneyResult;
}

/** Extract upsell opportunities. Returns null if analyzer didn't run. */
export function extractUpsell(report: PipelineReport): UpsellResult | null {
  const typed = report.analyzer_results?.upsell_opportunity;
  if (typed) return typed;
  const raw = report.results?.upsell_opportunity;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as UpsellResult;
}

/** Extract dialect sensitivity. Returns null if analyzer didn't run. */
export function extractDialect(report: PipelineReport): DialectSensitivityResult | null {
  const typed = report.analyzer_results?.dialect_sensitivity;
  if (typed) return typed;
  const raw = report.results?.dialect_sensitivity;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as DialectSensitivityResult;
}

/** Extract detailed compliance risk. Returns null if analyzer didn't run. */
export function extractComplianceRisk(report: PipelineReport): ComplianceRiskResult | null {
  const typed = report.analyzer_results?.compliance_risk;
  if (typed) return typed;
  const raw = report.results?.compliance_risk;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as ComplianceRiskResult;
}

/** Extract role classifier extended metrics. Returns null if not run. */
export function extractRoleMetrics(report: PipelineReport): RoleClassifierResult | null {
  const typed = report.analyzer_results?.role;
  if (typed) return typed;
  const raw = report.results?.role;
  if (!raw || typeof raw !== "object") return null;
  return raw as unknown as RoleClassifierResult;
}

/** Extract predictive risk details. Returns null if analyzer didn't run. */
export function extractPredictive(report: PipelineReport): PredictiveResult | null {
  const typed = report.analyzer_results?.predictive;
  if (typed) return typed;
  const risks = report.risks;
  if (!risks || typeof risks !== "object") return null;
  return risks as unknown as PredictiveResult;
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
  // Fas 5: typed analyzer outputs
  emotion: EmotionSegmentResult[];
  aspects: AspectItem[];
  trajectory: TrajectoryResult | null;
  rootCause: RootCauseResult | null;
  coaching: CoachingResult | null;
  customerEffort: CustomerEffortResult | null;
  activeListening: ActiveListeningResult | null;
  empathy: EmpathyResult | null;
  resolutionProbability: ResolutionProbabilityResult | null;
  journey: MultiTurnJourneyResult | null;
  upsell: UpsellResult | null;
  dialect: DialectSensitivityResult | null;
  complianceRisk: ComplianceRiskResult | null;
  roleMetrics: RoleClassifierResult | null;
  predictive: PredictiveResult | null;
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
    // Fas 5: typed analyzer outputs
    emotion: extractEmotion(report),
    aspects: extractAspects(report),
    trajectory: extractTrajectory(report),
    rootCause: extractRootCause(report),
    coaching: extractCoaching(report),
    customerEffort: extractCustomerEffort(report),
    activeListening: extractActiveListening(report),
    empathy: extractEmpathy(report),
    resolutionProbability: extractResolutionProbability(report),
    journey: extractJourney(report),
    upsell: extractUpsell(report),
    dialect: extractDialect(report),
    complianceRisk: extractComplianceRisk(report),
    roleMetrics: extractRoleMetrics(report),
    predictive: extractPredictive(report),
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

// ---------------------------------------------------------------------------
// Executive Insights aggregation (cross-call KPIs for management overview)
// ---------------------------------------------------------------------------

export interface RiskMetrics {
  churnRisk: number;
  escalationRisk: number;
  satisfactionScore: number;
  riskLevel: string;
}

export interface ExecutiveKpis {
  totalCalls: number;
  avgQaScore: number;
  avgSentiment: number;
  totalAlerts: number;
  qaPassRate: number;
  avgChurnRisk: number;
  avgEscalationRisk: number;
  avgSatisfaction: number;
  criticalCalls: number;
  totalLlmCostUsd: number;
}

export interface AgentBenchmark {
  agent: string;
  calls: number;
  avgQaScore: number;
  avgSentiment: number;
  avgEmpathy: number | null;
  alertCount: number;
  avgChurnRisk: number;
}

export interface ExecutiveSummary {
  kpis: ExecutiveKpis;
  agentBenchmarks: AgentBenchmark[];
  riskDistribution: Record<string, number>;
  topAlertRules: { ruleId: string; count: number }[];
  categoryBreakdown: { category: string; calls: number; avgSentiment: number; avgQa: number }[];
}

/** Extract risk metrics from a pipeline report. */
export function extractRiskMetrics(report: PipelineReport): RiskMetrics {
  const risks = report.risks as Record<string, unknown> | undefined;
  return {
    churnRisk: Number(risks?.churn_risk ?? 0),
    escalationRisk: Number(risks?.escalation_risk ?? 0),
    satisfactionScore: Number(risks?.satisfaction_score ?? 0.5),
    riskLevel: String(risks?.risk_level ?? "medium"),
  };
}

/** Aggregate executive-level KPIs across all demo reports. */
export function aggregateExecutiveSummary(calls: RealCall[]): ExecutiveSummary {
  if (calls.length === 0) {
    return {
      kpis: {
        totalCalls: 0,
        avgQaScore: 0,
        avgSentiment: 0,
        totalAlerts: 0,
        qaPassRate: 0,
        avgChurnRisk: 0,
        avgEscalationRisk: 0,
        avgSatisfaction: 0,
        criticalCalls: 0,
        totalLlmCostUsd: 0,
      },
      agentBenchmarks: [],
      riskDistribution: {},
      topAlertRules: [],
      categoryBreakdown: [],
    };
  }

  const rows = reportsToCallRows(calls);
  const qaScores = rows.map((r) => r.qaScore ?? 0);
  const sentiments = rows.map((r) => r.sentimentScore);
  const allAlerts = collectAllAlerts(calls);

  const riskMetrics = calls.map((c) => extractRiskMetrics(c.report));
  const avgChurnRisk = riskMetrics.reduce((s, r) => s + r.churnRisk, 0) / calls.length;
  const avgEscalationRisk = riskMetrics.reduce((s, r) => s + r.escalationRisk, 0) / calls.length;
  const avgSatisfaction = riskMetrics.reduce((s, r) => s + r.satisfactionScore, 0) / calls.length;
  const criticalCalls = riskMetrics.filter((r) => r.riskLevel === "critical").length;

  // LLM cost from llm_judge results
  let totalLlmCostUsd = 0;
  for (const c of calls) {
    const judge = extractLlmJudge(c.report);
    if (judge) totalLlmCostUsd += judge.total_cost_usd;
  }

  // Risk distribution
  const riskDistribution: Record<string, number> = {};
  for (const r of riskMetrics) {
    riskDistribution[r.riskLevel] = (riskDistribution[r.riskLevel] ?? 0) + 1;
  }

  // Top alert rules
  const ruleCounts = new Map<string, number>();
  for (const a of allAlerts) {
    ruleCounts.set(a.rule_id, (ruleCounts.get(a.rule_id) ?? 0) + 1);
  }
  const topAlertRules = [...ruleCounts.entries()]
    .map(([ruleId, count]) => ({ ruleId, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);

  // Agent benchmarks
  const byAgent = new Map<string, RealCall[]>();
  for (const call of calls) {
    const agent = call.transcript.meta.agent;
    const list = byAgent.get(agent) ?? [];
    list.push(call);
    byAgent.set(agent, list);
  }

  const agentBenchmarks: AgentBenchmark[] = [];
  for (const [agent, agentCalls] of byAgent) {
    const agentRows = reportsToCallRows(agentCalls);
    const agentRisks = agentCalls.map((c) => extractRiskMetrics(c.report));
    const empathyScores = agentCalls
      .map((c) => {
        const ap = c.report.results?.agent_performance as Record<string, Record<string, unknown>> | undefined;
        const agentPerf = ap?.agent;
        return typeof agentPerf?.empathy_score === "number" ? agentPerf.empathy_score : null;
      })
      .filter((v): v is number => v !== null);

    agentBenchmarks.push({
      agent,
      calls: agentCalls.length,
      avgQaScore: agentRows.reduce((s, r) => s + (r.qaScore ?? 0), 0) / agentRows.length,
      avgSentiment: agentRows.reduce((s, r) => s + r.sentimentScore, 0) / agentRows.length,
      avgEmpathy: empathyScores.length > 0 ? empathyScores.reduce((s, v) => s + v, 0) / empathyScores.length : null,
      alertCount: agentRows.reduce((s, r) => s + r.alertCount, 0),
      avgChurnRisk: agentRisks.reduce((s, r) => s + r.churnRisk, 0) / agentRisks.length,
    });
  }
  agentBenchmarks.sort((a, b) => b.avgQaScore - a.avgQaScore);

  // Category breakdown
  const byCategory = new Map<string, RealCall[]>();
  for (const call of calls) {
    const cat = call.transcript.meta.category;
    const list = byCategory.get(cat) ?? [];
    list.push(call);
    byCategory.set(cat, list);
  }
  const categoryBreakdown = [...byCategory.entries()].map(([category, catCalls]) => {
    const catRows = reportsToCallRows(catCalls);
    return {
      category,
      calls: catCalls.length,
      avgSentiment: catRows.reduce((s, r) => s + r.sentimentScore, 0) / catRows.length,
      avgQa: catRows.reduce((s, r) => s + (r.qaScore ?? 0), 0) / catRows.length,
    };
  });

  const qaPassRate = rows.filter((r) => r.qaPassed === true).length / rows.length;

  return {
    kpis: {
      totalCalls: calls.length,
      avgQaScore: qaScores.reduce((s, v) => s + v, 0) / qaScores.length,
      avgSentiment: sentiments.reduce((s, v) => s + v, 0) / sentiments.length,
      totalAlerts: allAlerts.length,
      qaPassRate,
      avgChurnRisk,
      avgEscalationRisk,
      avgSatisfaction,
      criticalCalls,
      totalLlmCostUsd,
    },
    agentBenchmarks,
    riskDistribution,
    topAlertRules,
    categoryBreakdown,
  };
}
