"use client";

import { useMemo } from "react";

import { useDemoReports } from "@/hooks/use-demo-reports";
import { collectAllAlerts, type AlertItem } from "@/lib/real-data";

export type AlertSeverityFilter = "all" | "critical" | "high" | "medium" | "low";

export interface UseAlertsResult {
  alerts: AlertItem[];
  totalCount: number;
  countsBySeverity: Record<string, number>;
  isLoading: boolean;
  isError: boolean;
}

/**
 * Collects all structured alerts across the demo pipeline reports
 * (extracted from `results.alerts` in each `/analyze_pipeline` response).
 * No extra API call needed — alerts are already in the cached reports.
 */
export function useAlerts(filter: AlertSeverityFilter = "all"): UseAlertsResult {
  const { reports, isLoading, isError } = useDemoReports();

  const allAlerts = useMemo(() => collectAllAlerts(reports), [reports]);

  const countsBySeverity = useMemo(() => {
    const counts: Record<string, number> = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
    for (const a of allAlerts) {
      counts[a.severity] = (counts[a.severity] ?? 0) + 1;
    }
    return counts;
  }, [allAlerts]);

  const filtered = useMemo(
    () => (filter === "all" ? allAlerts : allAlerts.filter((a) => a.severity === filter)),
    [allAlerts, filter],
  );

  return {
    alerts: filtered,
    totalCount: allAlerts.length,
    countsBySeverity,
    isLoading,
    isError,
  };
}
