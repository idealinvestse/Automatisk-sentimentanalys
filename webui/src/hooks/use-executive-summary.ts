"use client";

import { useMemo } from "react";

import { useDemoReports } from "@/hooks/use-demo-reports";
import { aggregateExecutiveSummary, type ExecutiveSummary } from "@/lib/real-data";

export interface UseExecutiveSummaryResult {
  summary: ExecutiveSummary | null;
  isLoading: boolean;
  isError: boolean;
  usingLiveData: boolean;
}

/**
 * Aggregates executive-level KPIs across cached pipeline reports.
 * No extra API call — all data comes from the cached `useDemoReports`
 * results (risks, qa, alerts, agent_performance, llm_judge).
 */
export function useExecutiveSummary(): UseExecutiveSummaryResult {
  const { reports, isLoading, isError, usingLiveData } = useDemoReports();

  const summary = useMemo(() => {
    if (reports.length === 0) return null;
    return aggregateExecutiveSummary(reports);
  }, [reports]);

  return { summary, isLoading, isError, usingLiveData };
}
