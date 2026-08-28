import type { Page } from "@playwright/test";

/**
 * Minimal pipeline stub so dashboard smoke tests do not hit a live API.
 * Analyzer-card tests use a richer mock in analyzer-cards.spec.ts.
 */
const SMOKE_PIPELINE = {
  sentiment_results: [{ label: "neutral", score: 0.1 }],
  intent_results: [{ intent: "other", confidence: 0.5 }],
  summary: { summary: "E2E smoke stub", action_items: [] },
  topics: { topics: [] },
  insights: {},
  risks: {
    churn_risk: 0.1,
    escalation_risk: 0.1,
    satisfaction_score: 0.5,
    risk_factors: [],
    risk_level: "low",
  },
  processing_time_s: 0.01,
  timestamp: new Date().toISOString(),
  llm: {},
  results: {
    qa: { overall_qa_score: 70, passed: true, criteria_results: [], compliance_flags: [] },
    alerts: [],
  },
  analyzer_results: null,
};

const EDGE_ANALYSIS = {
  profile: "callcenter",
  offline: true,
  llm_used: false,
  segments: [
    {
      text: "Tack för hjälpen",
      sentiment_label: "positiv",
      sentiment_score: 0.92,
      intent: "resolution",
    },
  ],
  summary: "Positiv kundrespons.",
  limitations: ["Ingen LLM"],
};

/** Stub health + auth ticket + common dashboard API routes. */
export async function stubDashboardApi(page: Page): Promise<void> {
  await page.route("**/health", (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: '{"status":"ok"}' }),
  );
  await page.route("**/ready", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: '{"status":"ok","ready":true}',
    }),
  );
  await page.route("**/ws/transcription/ticket", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ticket: "e2e-no-auth", expires_in: 300 }),
    }),
  );
  await page.route("**/api/backend/calls", (r) => {
    if (r.request().method() === "POST") {
      return r.fulfill({ status: 200, contentType: "application/json", body: "{}" });
    }
    return r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ calls: [], count: 0 }),
    });
  });
  await page.route("**/api/backend/calls?*", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ calls: [], count: 0 }),
    }),
  );
  await page.route("**/api/backend/calls/**", (r) =>
    r.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Call not found", error_code: "not_found" }),
    }),
  );
  await page.route("**/analyze_pipeline*", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(SMOKE_PIPELINE),
    }),
  );
  await page.route("**/analyze_pipeline/partial", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ...SMOKE_PIPELINE,
        results: { ...SMOKE_PIPELINE.results, partial: { incremental: true, reconciled: false } },
      }),
    }),
  );
  await page.route("**/analyze_pipeline/compare", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: ["fast", "balanced", "deep"],
        results: {},
        total_processing_time_s: 0.03,
        budget_exceeded: false,
        timestamp: new Date().toISOString(),
      }),
    }),
  );
  await page.route("**/insights/hot_topics", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ hot_topics: [], meta: {}, timestamp: new Date().toISOString() }),
    }),
  );
  await page.route("**/agent_performance/**", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        agent_id: "stub",
        metrics: {},
        cached: false,
        timestamp: new Date().toISOString(),
      }),
    }),
  );
  await page.route("**/alerts", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ alerts: [], timestamp: new Date().toISOString() }),
    }),
  );
  await page.route("**/alerting/status", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true, webhook: { circuit_breaker_open: false } }),
    }),
  );
  await page.route("**/search/semantic", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        query: "stub",
        hits: [],
        meta: {},
        timestamp: new Date().toISOString(),
      }),
    }),
  );
  await page.route("**/qa/score", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        qa: { overall_qa_score: 72, passed: true },
        timestamp: new Date().toISOString(),
      }),
    }),
  );
  await page.route("**/transcription/jobs**", (r) => {
    const url = r.request().url();
    if (url.includes("/cancel")) {
      return r.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: "stub",
          cancelled: true,
          timestamp: new Date().toISOString(),
        }),
      });
    }
    return r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ jobs: [], timestamp: new Date().toISOString() }),
    });
  });
  await page.route("**/upload", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        audio_path: "uploads/e2e.wav",
        filename: "e2e.wav",
        size_bytes: 8,
        timestamp: new Date().toISOString(),
      }),
    }),
  );
  await page.route("**/transcribe", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        transcript: {
          segments: [{ speaker: "Agent", text: "Hej", start: 0, end: 1 }],
        },
        timestamp: new Date().toISOString(),
      }),
    }),
  );
  await page.route("**/edge/analyze-text", (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(EDGE_ANALYSIS) }),
  );
  await page.route("**/edge/analyze-segments", (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(EDGE_ANALYSIS) }),
  );
}
