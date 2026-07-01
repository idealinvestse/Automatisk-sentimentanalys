/**
 * Mock/demo data for UI development, decoupled from the real data-source
 * decision (see docs/WEBUI_MODERNIZATION_PLAN.md, "Datakälla"-frågan).
 *
 * Shape mirrors `reports_to_table_rows()` in
 * app/nicegui_dashboard/services/demo_provider.py so swapping this out for
 * a real API response later is a drop-in replacement.
 */

export type SentimentLabel = "positive" | "neutral" | "negative";
export type RiskLevel = "low" | "medium" | "high" | "critical";

export interface CallRow {
  callId: string;
  title: string;
  agent: string;
  category: string;
  sentiment: SentimentLabel;
  sentimentScore: number;
  riskLevel: RiskLevel;
  alertCount: number;
  qaPassed: boolean | null;
  qaScore: number | null;
  durationS: number;
  timestamp: string;
}

export const MOCK_CALLS: CallRow[] = [
  {
    callId: "CALL-001",
    title: "Faktura fel – lyckad upplösning",
    agent: "Agent-Anna",
    category: "billing",
    sentiment: "positive",
    sentimentScore: 0.82,
    riskLevel: "low",
    alertCount: 0,
    qaPassed: true,
    qaScore: 94,
    durationS: 420,
    timestamp: "2026-06-30T08:12:00Z",
  },
  {
    callId: "CALL-002",
    title: "Leveransförsening – eskalering",
    agent: "Agent-Erik",
    category: "logistics",
    sentiment: "negative",
    sentimentScore: 0.21,
    riskLevel: "high",
    alertCount: 2,
    qaPassed: false,
    qaScore: 58,
    durationS: 610,
    timestamp: "2026-06-30T09:05:00Z",
  },
  {
    callId: "CALL-003",
    title: "Teknisk felsökning – nästan compliance-miss",
    agent: "Agent-Sara",
    category: "technical",
    sentiment: "neutral",
    sentimentScore: 0.52,
    riskLevel: "medium",
    alertCount: 1,
    qaPassed: true,
    qaScore: 76,
    durationS: 540,
    timestamp: "2026-06-30T09:40:00Z",
  },
  {
    callId: "CALL-004",
    title: "Fakturatvist – grundorsak identifierad",
    agent: "Agent-Anna",
    category: "billing",
    sentiment: "neutral",
    sentimentScore: 0.48,
    riskLevel: "medium",
    alertCount: 0,
    qaPassed: true,
    qaScore: 81,
    durationS: 480,
    timestamp: "2026-06-30T10:15:00Z",
  },
  {
    callId: "CALL-005",
    title: "Lyckad de-eskalering + lätt uppförsäljning",
    agent: "Agent-Johan",
    category: "sales",
    sentiment: "positive",
    sentimentScore: 0.9,
    riskLevel: "low",
    alertCount: 0,
    qaPassed: true,
    qaScore: 97,
    durationS: 390,
    timestamp: "2026-06-30T11:02:00Z",
  },
  {
    callId: "CALL-006",
    title: "Kontoproblem – lösning via chatt-överlämning",
    agent: "Agent-Erik",
    category: "account",
    sentiment: "negative",
    sentimentScore: 0.33,
    riskLevel: "medium",
    alertCount: 1,
    qaPassed: false,
    qaScore: 64,
    durationS: 355,
    timestamp: "2026-06-30T11:47:00Z",
  },
  {
    callId: "CALL-007",
    title: "Allmän produktfråga",
    agent: "Agent-Sara",
    category: "general",
    sentiment: "positive",
    sentimentScore: 0.71,
    riskLevel: "low",
    alertCount: 0,
    qaPassed: true,
    qaScore: 88,
    durationS: 210,
    timestamp: "2026-06-30T12:20:00Z",
  },
  {
    callId: "CALL-008",
    title: "Kritiskt GDPR-relaterat klagomål",
    agent: "Agent-Johan",
    category: "compliance",
    sentiment: "negative",
    sentimentScore: 0.12,
    riskLevel: "critical",
    alertCount: 3,
    qaPassed: false,
    qaScore: 41,
    durationS: 720,
    timestamp: "2026-06-30T13:05:00Z",
  },
];

export interface AgentSummary {
  agent: string;
  calls: number;
  avgSentiment: number;
  avgQaScore: number;
  alertCount: number;
}

export function summarizeAgents(calls: CallRow[] = MOCK_CALLS): AgentSummary[] {
  const byAgent = new Map<string, CallRow[]>();
  for (const call of calls) {
    const list = byAgent.get(call.agent) ?? [];
    list.push(call);
    byAgent.set(call.agent, list);
  }
  return Array.from(byAgent.entries())
    .map(([agent, rows]) => ({
      agent,
      calls: rows.length,
      avgSentiment: average(rows.map((r) => r.sentimentScore)),
      avgQaScore: average(rows.map((r) => r.qaScore ?? 0)),
      alertCount: rows.reduce((sum, r) => sum + r.alertCount, 0),
    }))
    .sort((a, b) => b.avgQaScore - a.avgQaScore);
}

export interface CategoryTrendPoint {
  category: string;
  calls: number;
  avgSentiment: number;
}

export function summarizeCategories(calls: CallRow[] = MOCK_CALLS): CategoryTrendPoint[] {
  const byCategory = new Map<string, CallRow[]>();
  for (const call of calls) {
    const list = byCategory.get(call.category) ?? [];
    list.push(call);
    byCategory.set(call.category, list);
  }
  return Array.from(byCategory.entries()).map(([category, rows]) => ({
    category,
    calls: rows.length,
    avgSentiment: average(rows.map((r) => r.sentimentScore)),
  }));
}

export function summarizeKpis(calls: CallRow[] = MOCK_CALLS) {
  const qaScores = calls.map((c) => c.qaScore).filter((v): v is number => v !== null);
  return {
    totalCalls: calls.length,
    avgSentiment: average(calls.map((c) => c.sentimentScore)),
    avgQaScore: average(qaScores),
    activeAlerts: calls.reduce((sum, c) => sum + c.alertCount, 0),
  };
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}
