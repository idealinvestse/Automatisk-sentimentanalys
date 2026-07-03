/**
 * Model routing tiers — mirrors src/llm/routing.py (RoutingTier + DEFAULT_MODELS).
 *
 * The frontend maps a tier to a concrete OpenRouter model slug and sends it
 * as `llm_model` in the /analyze_pipeline request body. The backend's
 * `select_model()` would normally do this, but since the demo transcripts
 * don't always trigger the LLM path, exposing the tier in the UI lets the
 * user explicitly choose cost/quality tradeoff for Testlabb runs.
 */

export type RoutingTier = "fast" | "balanced" | "deep";

export interface TierInfo {
  id: RoutingTier;
  label: string;
  description: string;
  model: string;
  /** Approximate cost per 1M tokens (USD) — from OpenRouter catalog defaults. */
  costPerMTokens: number;
  /** Relative quality indicator (1-5). */
  quality: number;
  /** Expected latency band. */
  latency: "low" | "medium" | "high";
}

export const ROUTING_TIERS: Record<RoutingTier, TierInfo> = {
  fast: {
    id: "fast",
    label: "FAST",
    description: "Snabb & billig — mistral-small. Bra för korta samtal och batch.",
    model: "mistralai/mistral-small-3.1-24b-instruct",
    costPerMTokens: 0.15,
    quality: 3,
    latency: "low",
  },
  balanced: {
    id: "balanced",
    label: "BALANCED",
    description: "Standard — mistral-medium. Bra balans mellan kostnad och kvalitet.",
    model: "mistralai/mistral-medium-3.5",
    costPerMTokens: 0.40,
    quality: 4,
    latency: "medium",
  },
  deep: {
    id: "deep",
    label: "DEEP",
    description: "Max kvalitet — mistral-large. För komplexa/långa samtal och djupanalys.",
    model: "mistralai/mistral-large-2512",
    costPerMTokens: 2.0,
    quality: 5,
    latency: "high",
  },
};

export const TIER_LIST: TierInfo[] = [ROUTING_TIERS.fast, ROUTING_TIERS.balanced, ROUTING_TIERS.deep];

/** Map a tier to the llm_model slug to send to the backend. */
export function tierToModel(tier: RoutingTier): string {
  return ROUTING_TIERS[tier].model;
}

/**
 * Resolve the effective tier given segment count and flags — mirrors the
 * override logic in routing.py:select_model() (deep_analysis or >=20 segments
 * forces DEEP; <6 segments forces FAST).
 */
export function resolveEffectiveTier(
  tier: RoutingTier,
  segmentCount: number,
  deepAnalysis: boolean,
): RoutingTier {
  if (deepAnalysis || segmentCount >= 20) return "deep";
  if (segmentCount < 6) return "fast";
  return tier;
}
