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
