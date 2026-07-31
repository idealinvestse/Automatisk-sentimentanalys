/**
 * Analysis perspectives — selectable paid-model profiles with cost optimization.
 * Backend SSOT: GET /llm/analysis-profiles (src/llm/paid_model_advisor.py)
 */

export type AnalysisPerspectiveId =
  | "cost_saver"
  | "batch_throughput"
  | "sentiment_refine"
  | "intent_routing"
  | "root_cause"
  | "coaching_qa"
  | "compliance_risk"
  | "summary_actions"
  | "swedish_quality"
  | "holistic_deep"
  | "balanced_ops"
  | "premium_reasoning";

export interface AnalysisProfileMenuItem {
  id: string;
  label: string;
  description: string;
  use_when?: string;
  icon?: string;
  cost_priority?: number;
  quality_priority?: number;
  model?: string | null;
  provider?: string | null;
  blended_usd_per_m?: number | null;
  est_cost_per_call_usd?: number | null;
  selectable?: {
    provider?: string;
    llm_model?: string | null;
    analysis_perspective?: string;
    use_mistral_llm?: boolean;
    deep_analysis?: boolean;
  };
}

export interface AnalysisProfilesResponse {
  generated_at?: string;
  candidate_count?: number;
  cached?: boolean;
  menu?: AnalysisProfileMenuItem[];
  profiles?: Array<Record<string, unknown>>;
  notes?: Record<string, string>;
  providers_configured?: Record<string, boolean>;
}

/** Fallback menu if API is offline — still selectable with sensible defaults. */
export const FALLBACK_ANALYSIS_MENU: AnalysisProfileMenuItem[] = [
  {
    id: "balanced_ops",
    label: "Balanserad drift",
    description: "Standardval: bra kvalitet till rimlig kostnad.",
    use_when: "Default för daglig drift",
    provider: "openrouter",
    model: "mistralai/mistral-medium-3-5",
    blended_usd_per_m: 5.1,
    est_cost_per_call_usd: 0.017,
    selectable: {
      provider: "openrouter",
      llm_model: "mistralai/mistral-medium-3-5",
      analysis_perspective: "balanced_ops",
      use_mistral_llm: true,
    },
  },
  {
    id: "cost_saver",
    label: "Kostnadssparare",
    description: "Billigast paid-modell som klarar svensk text.",
    use_when: "Batch, nattjobb",
    provider: "openrouter",
    model: "mistralai/mistral-nemo",
    blended_usd_per_m: 0.026,
    selectable: {
      provider: "openrouter",
      llm_model: "mistralai/mistral-nemo",
      analysis_perspective: "cost_saver",
      use_mistral_llm: true,
    },
  },
  {
    id: "coaching_qa",
    label: "QA / coachning",
    description: "Agentbedömning och konkreta coachningspunkter.",
    use_when: "Kvalitetssäkring",
    provider: "openrouter",
    model: "mistralai/mistral-large-2512",
    selectable: {
      provider: "openrouter",
      llm_model: "mistralai/mistral-large-2512",
      analysis_perspective: "coaching_qa",
      use_mistral_llm: true,
      deep_analysis: true,
    },
  },
  {
    id: "holistic_deep",
    label: "Holistisk djupanalys",
    description: "Full call-analys i ett svep.",
    use_when: "VIP / svåra cases",
    provider: "openrouter",
    model: "mistralai/mistral-large-2512",
    selectable: {
      provider: "openrouter",
      llm_model: "mistralai/mistral-large-2512",
      analysis_perspective: "holistic_deep",
      use_mistral_llm: true,
      deep_analysis: true,
    },
  },
];

export function formatUsd(n: number | null | undefined, digits = 3): string {
  if (n == null || Number.isNaN(n)) return "—";
  if (n < 0.001) return `$${n.toFixed(5)}`;
  return `$${n.toFixed(digits)}`;
}

export function costQualityLabel(costP?: number, qualP?: number): string {
  const c = costP ?? 0.5;
  const q = qualP ?? 0.5;
  if (c >= 0.7) return "Kostnad först";
  if (q >= 0.8) return "Kvalitet först";
  return "Balanserad";
}
