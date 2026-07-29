import { test, expect, type Page } from "@playwright/test";

/**
 * E2E tests for the Fas 5 analyzer-cards UI.
 *
 * Mocks `/analyze_pipeline` responses with realistic `analyzer_results` data
 * so the tests run in isolation without requiring the backend pipeline.
 */

// ---------------------------------------------------------------------------
// Mock data — a single rich pipeline response with all analyzer outputs
// ---------------------------------------------------------------------------

const MOCK_PIPELINE_RESPONSE = {
  sentiment_results: [
    { label: "neutral", score: 0.5 },
    { label: "negativ", score: -0.6 },
    { label: "positiv", score: 0.7 },
  ],
  intent_results: [
    { intent: "greeting", confidence: 0.9 },
    { intent: "complaint", confidence: 0.8 },
    { intent: "resolution", confidence: 0.85 },
  ],
  summary: {
    summary: "Kund ringer angående felaktig faktura. Agent krediterar och löser problemet.",
    action_items: ["Skicka rättad faktura", "Följ upp om 3 dagar"],
  },
  topics: { topics: [{ label: "fakturering", weight: 0.8 }] },
  insights: { key_findings: ["Faktureringsfel vanligt"] },
  risks: {
    churn_risk: 0.3,
    escalation_risk: 0.2,
    satisfaction_score: 0.7,
    risk_factors: ["billing_error"],
    risk_level: "low",
  },
  processing_time_s: 1.23,
  timestamp: new Date().toISOString(),
  llm: {},
  results: {
    qa: { overall_qa_score: 78, passed: true, criteria_results: [], compliance_flags: [] },
    alerts: [],
    emotion: [
      { primary: "frustration", scores: { frustration: 0.8, neutral: 0.2 }, speaker: "Kund" },
      { primary: "tillfredsställelse", scores: { tillfredsställelse: 0.9 }, speaker: "Kund" },
    ],
    aspect: [
      { aspect: "fakturering_pris", sentiment: "negative", score: -0.6, evidence: "felaktig faktura", speaker: "Kund" },
      { aspect: "agent_attityd", sentiment: "positive", score: 0.8, evidence: "snabb lösning", speaker: "Kund" },
    ],
    trajectory: {
      customer_sentiment_slope: 0.15,
      escalation_events: 1,
      escalation_event_details: [{ turn: 1, type: "sentiment_drop", delta: -0.4, evidence: "jag har fått en felaktig faktura" }],
      peak_frustration_turn: 1,
      sentiment_trend: [0.5, -0.6, 0.7],
    },
    root_cause: {
      root_causes: [{ cause: "produktfel", count: 2, recommendation: "Prioritera buggfix" }],
      top_root_cause: "produktfel",
      evidence_examples: [{ evidence: "felaktig faktura" }],
      overall_risk: "medium",
    },
    actionable_coaching: {
      coaching_insights: [
        { rule_id: "low_empathy", priority: "high", recommendation: "Träna validerande fraser", evidence: 45 },
      ],
      top_recommendation: "Träna validerande fraser",
      insight_count: 1,
    },
    customer_effort: {
      overall_ces: 35,
      scale: "0-100 (högre = mer effort)",
      per_segment: [{ speaker: "Kund", start: 0, end: 10, effort_score: 30 }],
      coaching_tips: ["Förenkla språk"],
    },
    active_listening: {
      listening_score: 72,
      backchannel_count: 3,
      speaker_balance: { Agent: 60, Kund: 40 },
      events: [{ type: "backchannel", speaker: "Agent", time: 5 }],
      tips: [],
    },
    empathy: {
      overall_empathy: 65,
      scale: "0-100",
      per_segment: [{ speaker: "Agent", start: 18, empathy_score: 70, evidence: ["jag förstår"] }],
      coaching_tips: [],
    },
    resolution_probability: {
      resolution_probability: 80,
      confidence: 65,
      recommended_action: "Sammanfatta och bekräfta lösning",
      factors: { sentiment_trend: "positive", customer_effort_impact: 35, empathy_impact: 65 },
    },
    multi_turn_journey: {
      journey_stages: [
        { stage: "opening", start: 0, speaker: "Agent", text_snippet: "Hej", intent: "greeting", sentiment: "neutral" },
        { stage: "resolution", start: 42, speaker: "Agent", text_snippet: "krediterar", intent: "resolution", sentiment: "positive" },
      ],
      resolved: true,
      unresolved_count: 0,
      key_turning_points: [],
      recommendation: "Bra journey",
    },
    upsell_opportunity: {
      opportunities: [{ speaker: "Kund", start: 65, end: 75, confidence: 60, signals: ["positive_context"], suggested_action: "Erbjud tillägg", evidence: "tack" }],
      count: 1,
      recommendation: "Träna agenter",
    },
    compliance_risk: {
      overall_risk_level: "low",
      flagged_segments: [],
      recommendation: "Inga problem",
    },
    role: {
      roles: { Agent: "agent", Kund: "customer" },
      talk_ratios: { agent: 60, customer: 40, talk_listen_ratio: 1.5 },
      question_density: { agent: 0.2, customer: 0.1 },
      lexical_formality: 0.7,
      intervention_count: 1,
      sentiment_variance: 0.3,
      num_agent_turns: 2,
      num_customer_turns: 2,
    },
    predictive: {
      churn_risk: 0.3,
      escalation_risk: 0.2,
      satisfaction_score: 0.7,
      risk_factors: ["billing_error"],
      risk_level: "low",
      recommended_action: null,
    },
  },
  analyzer_results: null, // will be set below to match the typed shape
};

// The typed analyzer_results view (mirrors what build_analyzer_results produces)
MOCK_PIPELINE_RESPONSE.analyzer_results = {
  emotion: MOCK_PIPELINE_RESPONSE.results.emotion,
  aspect: MOCK_PIPELINE_RESPONSE.results.aspect,
  trajectory: MOCK_PIPELINE_RESPONSE.results.trajectory,
  root_cause: MOCK_PIPELINE_RESPONSE.results.root_cause,
  actionable_coaching: MOCK_PIPELINE_RESPONSE.results.actionable_coaching,
  customer_effort: MOCK_PIPELINE_RESPONSE.results.customer_effort,
  active_listening: MOCK_PIPELINE_RESPONSE.results.active_listening,
  empathy: MOCK_PIPELINE_RESPONSE.results.empathy,
  resolution_probability: MOCK_PIPELINE_RESPONSE.results.resolution_probability,
  multi_turn_journey: MOCK_PIPELINE_RESPONSE.results.multi_turn_journey,
  upsell_opportunity: MOCK_PIPELINE_RESPONSE.results.upsell_opportunity,
  dialect_sensitivity: null,
  compliance_risk: MOCK_PIPELINE_RESPONSE.results.compliance_risk,
  role: MOCK_PIPELINE_RESPONSE.results.role,
  predictive: MOCK_PIPELINE_RESPONSE.results.predictive,
  agent_performance: null,
  qa: MOCK_PIPELINE_RESPONSE.results.qa,
  agent_assessment: null,
  agent_assessment_local: null,
  customer_metrics: null,
  pii_redaction: null,
  alerts: [],
  llm_judge: null,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function mockBackend(page: Page) {
  // Stub health check + auth ticket probe (connectionStatus)
  await page.route("**/health", (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: '{"status":"ok"}' }),
  );
  await page.route("**/ws/transcription/ticket", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ticket: "e2e-no-auth", expires_in: 300 }),
    }),
  );
  // Stub /analyze_pipeline — return the same rich response for any request body
  await page.route("**/analyze_pipeline", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_PIPELINE_RESPONSE),
    }),
  );
  // Stub Fas 4 aggregate endpoints (return empty but valid shapes)
  await page.route("**/insights/hot_topics", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ hot_topics: [], meta: {}, timestamp: new Date().toISOString() }),
    }),
  );
  await page.route("**/alerts", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ alerts: [], timestamp: new Date().toISOString() }),
    }),
  );
}

// ---------------------------------------------------------------------------
// Tests: /analysis page
// ---------------------------------------------------------------------------

test.describe("/analysis page — analyzer cards", () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  test("renders call selector and analyzer cards", async ({ page }) => {
    await page.goto("/analysis");
    // Wait for the call selector to appear
    await expect(page.getByRole("heading", { name: "Analysdetaljer" })).toBeVisible();
    await expect(page.getByText("Välj samtal")).toBeVisible();

    // Wait for the mock data to load and cards to render.
    // Use heading role to avoid strict-mode violations from description text.
    // EmotionCard
    await expect(page.getByRole("heading", { name: "Känslolabels" })).toBeVisible({ timeout: 15000 });
    await expect(page.getByText("Frustration", { exact: true })).toBeVisible();

    // AspectCard
    await expect(page.getByRole("heading", { name: /Aspektbaserad Sentiment/i })).toBeVisible();
    await expect(page.getByText("fakturering pris")).toBeVisible();

    // TrajectoryCard
    await expect(page.getByRole("heading", { name: "Samtalstrajectory" })).toBeVisible();
    await expect(page.getByText("Förbättras")).toBeVisible();

    // RootCauseCard
    await expect(page.getByRole("heading", { name: "Rotorsaksanalys" })).toBeVisible();
    await expect(page.getByText("produktfel").first()).toBeVisible();

    // CoachingCard
    await expect(page.getByRole("heading", { name: "Coaching-rekommendationer" })).toBeVisible();
    await expect(page.getByText("Träna validerande fraser").first()).toBeVisible();

    // CustomerEffortCard
    await expect(page.getByRole("heading", { name: "Kundinsats (CES)" })).toBeVisible();

    // ActiveListeningCard
    await expect(page.getByRole("heading", { name: "Aktivt lyssnande" })).toBeVisible();

    // EmpathyCard
    await expect(page.getByRole("heading", { name: "Empati", exact: true })).toBeVisible();

    // ResolutionProbabilityCard
    await expect(page.getByRole("heading", { name: "Lösnings sannolikhet" })).toBeVisible();

    // JourneyCard
    await expect(page.getByRole("heading", { name: "Kundens resa (multi-turn)" })).toBeVisible();

    // UpsellCard
    await expect(page.getByRole("heading", { name: "Upsell-möjligheter" })).toBeVisible();

    // RoleMetricsCard
    await expect(page.getByRole("heading", { name: "Rollanalys" })).toBeVisible();

    // PredictiveCard
    await expect(page.getByRole("heading", { name: "Prediktiv risk" })).toBeVisible();

    // SummaryCard
    await expect(page.getByRole("heading", { name: "Sammanfattning" })).toBeVisible();
  });

  test("no console errors on /analysis", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/analysis");
    await expect(page.getByRole("heading", { name: "Analysdetaljer" })).toBeVisible();
    // Wait for cards to render
    await expect(page.getByRole("heading", { name: "Känslolabels" })).toBeVisible({ timeout: 15000 });
    expect(errors, `Console errors: ${errors.join("; ")}`).toEqual([]);
  });

  test("journey card shows resolved badge", async ({ page }) => {
    await page.goto("/analysis");
    await expect(page.getByRole("heading", { name: "Kundens resa (multi-turn)" })).toBeVisible({ timeout: 15000 });
    await expect(page.getByText("Löst", { exact: true })).toBeVisible();
  });

  test("trajectory sentiment trend bar chart renders", async ({ page }) => {
    await page.goto("/analysis");
    await expect(page.getByText("Sentiment över tid")).toBeVisible({ timeout: 15000 });
    // The trend bar chart has 3 bars (matching sentiment_trend length)
    const bars = page.locator(".bg-success\\/70, .bg-destructive\\/70, .bg-muted");
    await expect(bars.first()).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// Tests: /calls/[id] page — analyzer cards in detail view
// ---------------------------------------------------------------------------

test.describe("/calls/[id] page — analyzer cards", () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  test("renders analyzer cards section", async ({ page }) => {
    await page.goto("/calls/CALL-001");
    // Wait for the page to load
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });
    // The "Analysdetaljer" section header should appear
    await expect(page.getByText("Analysdetaljer").first()).toBeVisible({ timeout: 15000 });
    // EmotionCard should render (use heading role to avoid strict-mode violation)
    await expect(page.getByRole("heading", { name: "Känslolabels" })).toBeVisible();
  });

  test("no console errors on call detail", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/calls/CALL-001");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });
    await expect(page.getByRole("heading", { name: "Känslolabels" })).toBeVisible({ timeout: 15000 });
    expect(errors, `Console errors: ${errors.join("; ")}`).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Tests: Overview page — new aggregator KPIs
// ---------------------------------------------------------------------------

test.describe("Overview page — Fas 5 aggregator KPIs", () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  test("renders CES, coaching, upsell and resolution KPIs", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Översikt" })).toBeVisible();
    // New KPI cards
    await expect(page.getByText("Snitt CES")).toBeVisible({ timeout: 15000 });
    await expect(page.getByText("Coaching-insikter")).toBeVisible();
    await expect(page.getByText("Upsell-möjligheter")).toBeVisible();
    await expect(page.getByText("Snitt lösningsgrad")).toBeVisible();
  });

  test("no console errors on overview", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/");
    await expect(page.getByText("Snitt CES")).toBeVisible({ timeout: 15000 });
    expect(errors, `Console errors: ${errors.join("; ")}`).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Tests: Testlab page — typed analyzer output summary
// ---------------------------------------------------------------------------

test.describe("Testlab page — analyzer output summary", () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  test("renders analyzer badges after running pipeline", async ({ page }) => {
    await page.goto("/testlab");
    await expect(page.getByRole("heading", { name: "Testlabb" })).toBeVisible();

    // Enter minimal segments and run
    const textarea = page.locator("textarea");
    await textarea.fill(JSON.stringify([{ text: "Hej", speaker: "Agent" }]));
    await page.getByRole("button", { name: /Analysera/i }).click();

    // Wait for the "Analys klar" badge (exact match to avoid toast notification)
    await expect(page.getByText("Analys klar", { exact: true })).toBeVisible({ timeout: 15000 });

    // The typed analyzer output summary should show badges
    await expect(page.getByText("Analyzer-resultat (typed view)")).toBeVisible();
    // At least the emotion badge should appear
    await expect(page.getByText("emotion", { exact: false }).first()).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// Tests: Sidebar navigation to /analysis
// ---------------------------------------------------------------------------

test("sidebar has Analysdetaljer link", async ({ page }) => {
  await page.route("**/health", (r) =>
    r.fulfill({ status: 200, contentType: "application/json", body: '{"status":"ok"}' }),
  );
  await page.goto("/");
  // The sidebar is hidden on small screens (md:flex), so ensure viewport is wide enough
  await page.setViewportSize({ width: 1280, height: 720 });
  const link = page.getByRole("link", { name: /Analysdetaljer/i });
  await expect(link).toBeVisible();
  await link.click();
  await page.waitForURL(/\/analysis/);
  await expect(page).toHaveURL(/\/analysis/);
});
