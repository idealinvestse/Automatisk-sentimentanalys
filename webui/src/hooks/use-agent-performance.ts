"use client";

import { useQueries } from "@tanstack/react-query";

import { apiClient, type AgentPerformanceResponse } from "@/lib/api/client";
import type { RealCall } from "@/lib/real-data";
import { summarizeAgents, type CallRow } from "@/lib/mock-data";

export interface AgentPerformanceRow {
  agent: string;
  calls: number;
  /** 0..1, derived from this agent's CallRows (always available). */
  avgSentiment: number;
  /** 0..100, derived from this agent's CallRows (always available). */
  avgQaScore: number;
  alertCount: number;
  /** 0..1 empathy score from the Fas 4 /agent_performance aggregate, when available. */
  empathyScore: number | null;
  /** Count of compliance flags across the agent's calls (the aggregate endpoint returns a count, not the flag text). */
  complianceFlagCount: number;
  /** Whether empathyScore/complianceFlagCount came from the API or are unavailable (local-only). */
  source: "api" | "local";
}

/**
 * Fetches real aggregate agent metrics via `POST /agent_performance/{agent_id}`
 * (Fas 4) for every unique agent found in `calls`/`reports`, one request per
 * agent metrics from the FastAPI backend.
 *
 * Per-agent call count / sentiment / QA score are always derived locally from
 * the already-fetched CallRows (cheap, always available); only the
 * empathy/compliance breakdown depends on the extra aggregate call and
 * degrades gracefully to "local" (no empathy data) if that call fails.
 */
export function useAgentPerformance(calls: CallRow[], reports: RealCall[]) {
  const localSummaries = summarizeAgents(calls);
  const agentIds = localSummaries.map((s) => s.agent);

  const queries = useQueries({
    queries: agentIds.map((agentId) => {
      const segmentsList = reports
        .filter((r) => r.transcript.meta.agent === agentId)
        .map((r) => r.transcript.segments);
      return {
        queryKey: ["agent_performance", agentId, segmentsList.length],
        queryFn: () => apiClient.getAgentPerformance<AgentPerformanceResponse>(agentId, segmentsList),
        enabled: segmentsList.length > 0,
        staleTime: 5 * 60_000,
        retry: 1,
      };
    }),
  });

  const isLoading = queries.some((q) => q.isLoading);

  const rows: AgentPerformanceRow[] = localSummaries.map((local, i) => {
    const api = queries[i]?.data?.metrics;
    const empathyScore = api?.averages?.empathy_score;
    return {
      agent: local.agent,
      calls: local.calls,
      avgSentiment: local.avgSentiment,
      avgQaScore: local.avgQaScore,
      alertCount: local.alertCount,
      empathyScore: typeof empathyScore === "number" ? empathyScore : null,
      complianceFlagCount: api?.total_compliance_flags ?? 0,
      source: api ? "api" : "local",
    };
  });

  return { rows, isLoading };
}
