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

// ---------------------------------------------------------------------------
// Call detail (Fas 2): transcript, QA scorecard, evidence, emotion timeline.
// Mirrors qa_scorecard.py / evidence_panel.py / emotion_timeline.py shapes.
// Only a subset of MOCK_CALLS has detail data, to also exercise the
// EmptyState path for calls without a deep-dive report.
// ---------------------------------------------------------------------------

export interface TranscriptTurn {
  speaker: "Agent" | "Kund";
  text: string;
  start: number;
}

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

export interface EmotionPoint {
  t: number;
  score: number;
}

export interface CallDetail {
  callId: string;
  transcript: TranscriptTurn[];
  qa: CallQa;
  evidenceQuotes: string[];
  emotionTimeline: EmotionPoint[];
}

export const MOCK_CALL_DETAILS: Record<string, CallDetail> = {
  "CALL-001": {
    callId: "CALL-001",
    transcript: [
      { speaker: "Agent", start: 0, text: "Hej, jag heter Anna på kundtjänst, hur kan jag hjälpa dig idag?" },
      { speaker: "Kund", start: 8, text: "Hej Anna, jag har fått en faktura på 890 kr som jag inte förstår." },
      { speaker: "Agent", start: 18, text: "Tack för att du ringer in. Kan jag få ditt kundnummer så kollar jag upp det direkt?" },
      { speaker: "Kund", start: 32, text: "Ja, det är 19851203-1234. Jag har aldrig ringt utomlands." },
      { speaker: "Agent", start: 42, text: "Jag ser felet nu — det var en systembugg. Jag krediterar beloppet direkt." },
      { speaker: "Kund", start: 55, text: "Åh, tack så mycket! Det var snabbt löst." },
    ],
    qa: {
      score: 94,
      passed: true,
      riskLevel: "low",
      complianceFlags: [],
      criteria: [
        { criterion: "Hälsningsfras", passed: true, score: 100, evidence: "\"Hej, jag heter Anna...\"" },
        { criterion: "Verifierade kundidentitet", passed: true, score: 100, evidence: "\"kundnummer... 19851203-1234\"" },
        { criterion: "Löste ärendet", passed: true, score: 90, evidence: "\"Jag krediterar beloppet direkt.\"" },
        { criterion: "Avslutsfras", passed: true, score: 85, evidence: "\"Tack så mycket!\"" },
      ],
    },
    evidenceQuotes: [
      "\"Jag förstår att det känns frustrerande.\"",
      "\"Jag ser felet nu — det var en systembugg.\"",
    ],
    emotionTimeline: [
      { t: 0, score: 0.5 },
      { t: 15, score: 0.35 },
      { t: 30, score: 0.4 },
      { t: 45, score: 0.75 },
      { t: 60, score: 0.9 },
    ],
  },
  "CALL-002": {
    callId: "CALL-002",
    transcript: [
      { speaker: "Kund", start: 0, text: "Min leverans är fem dagar sen och ingen har hört av sig!" },
      { speaker: "Agent", start: 10, text: "Jag beklagar det verkligen. Låt mig kolla spårningen." },
      { speaker: "Kund", start: 20, text: "Det här är tredje gången jag ringer om samma paket." },
      { speaker: "Agent", start: 30, text: "Jag ser det, och det är inte okej. Jag eskalerar till logistikteamet nu." },
      { speaker: "Kund", start: 45, text: "Jag vill ha kompensation för besväret." },
    ],
    qa: {
      score: 58,
      passed: false,
      riskLevel: "high",
      complianceFlags: ["Ingen kompensation erbjuden", "Lång väntetid innan eskalering"],
      criteria: [
        { criterion: "Hälsningsfras", passed: true, score: 80, evidence: "Standardhälsning användes." },
        { criterion: "Empati vid klagomål", passed: true, score: 70, evidence: "\"Jag beklagar det verkligen.\"" },
        { criterion: "Erbjöd kompensation", passed: false, score: 20, evidence: "Ingen kompensation nämndes i samtalet." },
        { criterion: "Löste ärendet inom samtalet", passed: false, score: 30, evidence: "Eskalerades utan löst datum." },
      ],
    },
    evidenceQuotes: [
      "\"Det här är tredje gången jag ringer om samma paket.\"",
      "\"Jag vill ha kompensation för besväret.\"",
    ],
    emotionTimeline: [
      { t: 0, score: 0.15 },
      { t: 15, score: 0.2 },
      { t: 30, score: 0.1 },
      { t: 45, score: 0.05 },
    ],
  },
  "CALL-008": {
    callId: "CALL-008",
    transcript: [
      { speaker: "Kund", start: 0, text: "Jag vill veta varför ni delat mina uppgifter med tredje part utan samtycke." },
      { speaker: "Agent", start: 12, text: "Det ska absolut inte ha skett. Kan du berätta mer om vad du sett?" },
      { speaker: "Kund", start: 25, text: "Jag fick reklam från ett företag jag aldrig varit i kontakt med." },
      { speaker: "Agent", start: 40, text: "Jag loggar detta som ett GDPR-ärende och eskalerar direkt till dataskyddsombudet." },
    ],
    qa: {
      score: 41,
      passed: false,
      riskLevel: "critical",
      complianceFlags: ["Möjlig GDPR-överträdelse", "Kräver DPO-eskalering", "Ej löst inom SLA"],
      criteria: [
        { criterion: "Korrekt eskaleringsväg", passed: true, score: 90, evidence: "\"eskalerar direkt till dataskyddsombudet.\"" },
        { criterion: "Undvek att lova utfall", passed: false, score: 10, evidence: "Inga tydliga nästa-steg kommunicerades till kund." },
        { criterion: "Dokumenterade ärendet korrekt", passed: false, score: 25, evidence: "Ofullständig ärendenotering." },
      ],
    },
    evidenceQuotes: [
      "\"Jag vill veta varför ni delat mina uppgifter med tredje part utan samtycke.\"",
      "\"Jag loggar detta som ett GDPR-ärende...\"",
    ],
    emotionTimeline: [
      { t: 0, score: 0.1 },
      { t: 15, score: 0.08 },
      { t: 30, score: 0.12 },
      { t: 45, score: 0.15 },
    ],
  },
};

export interface HotTopic {
  topic: string;
  mentions: number;
  avgSentiment: number;
  trend: "up" | "down" | "flat";
}

export const MOCK_HOT_TOPICS: HotTopic[] = [
  { topic: "Leveransförsening", mentions: 24, avgSentiment: 0.22, trend: "up" },
  { topic: "Fakturafel", mentions: 18, avgSentiment: 0.55, trend: "flat" },
  { topic: "GDPR / dataskydd", mentions: 6, avgSentiment: 0.15, trend: "up" },
  { topic: "Kontoåterställning", mentions: 15, avgSentiment: 0.61, trend: "down" },
  { topic: "Produktfrågor", mentions: 31, avgSentiment: 0.7, trend: "flat" },
];
