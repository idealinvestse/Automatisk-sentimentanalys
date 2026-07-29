"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { apiClient, type AlertsResponse } from "@/lib/api/client";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { isApiConnected, useHealth } from "@/hooks/use-health";
import { collectAllAlerts, type AlertItem, type AlertSeverity } from "@/lib/real-data";

export type AlertSeverityFilter = "all" | "critical" | "high" | "medium" | "low";

export interface UseAlertsResult {
  alerts: AlertItem[];
  totalCount: number;
  countsBySeverity: Record<string, number>;
  isLoading: boolean;
  isError: boolean;
  /** pipeline = embedded in analyze_pipeline; api = POST /alerts */
  source: "pipeline" | "api";
}

function normalizeSeverity(s: unknown): AlertSeverity {
  const v = String(s ?? "info").toLowerCase();
  if (v === "critical" || v === "high" || v === "medium" || v === "low") return v;
  return "info";
}

function mapServerAlerts(raw: Record<string, unknown>[]): AlertItem[] {
  return raw.map((a) => ({
    rule_id: String(a?.rule_id ?? "unknown"),
    severity: normalizeSeverity(a?.severity),
    message: String(a?.message ?? ""),
    evidence_spans: Array.isArray(a?.evidence_spans)
      ? (a.evidence_spans as AlertItem["evidence_spans"])
      : [],
    recommended_actions: Array.isArray(a?.recommended_actions)
      ? (a.recommended_actions as string[])
      : [],
    triggered_values: (a?.triggered_values as Record<string, unknown>) ?? {},
    source: a?.source ? String(a.source) : "api",
    callId: typeof a?.call_id === "string" ? a.call_id : undefined,
    callTitle: typeof a?.call_title === "string" ? a.call_title : undefined,
    agent: typeof a?.agent === "string" ? a.agent : undefined,
  }));
}

/**
 * Alerts for Insights: prefer POST /alerts (Fas 4) when API is connected and
 * we have segments; fall back to alerts embedded in pipeline reports.
 */
export function useAlerts(filter: AlertSeverityFilter = "all"): UseAlertsResult {
  const { reports, isLoading: reportsLoading, isError: reportsError } = useDemoReports();
  const { data: health } = useHealth();
  const apiOk = isApiConnected(health);

  const segmentsList = useMemo(
    () => reports.map((r) => r.transcript.segments),
    [reports],
  );

  const serverQuery = useQuery({
    queryKey: ["alerts", "server", segmentsList.length, reports.map((r) => r.transcript.id).join(",")],
    queryFn: () => apiClient.getAlerts(segmentsList),
    enabled: apiOk && segmentsList.length > 0,
    staleTime: 60_000,
    retry: 1,
  });

  const pipelineAlerts = useMemo(() => collectAllAlerts(reports), [reports]);

  const serverAlerts = useMemo(() => {
    const data = serverQuery.data as AlertsResponse | undefined;
    if (!data?.alerts?.length) return null;
    return mapServerAlerts(data.alerts);
  }, [serverQuery.data]);

  const useApi = Boolean(serverAlerts && !serverQuery.isError);
  const allAlerts = useApi && serverAlerts ? serverAlerts : pipelineAlerts;

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
    isLoading: reportsLoading || (apiOk && segmentsList.length > 0 && serverQuery.isLoading),
    isError: reportsError && serverQuery.isError,
    source: useApi ? "api" : "pipeline",
  };
}
