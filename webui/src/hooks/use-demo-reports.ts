"use client";

import { useQueries } from "@tanstack/react-query";

import { apiClient, type PipelineReport } from "@/lib/api/client";
import { DEMO_TRANSCRIPTS, type DemoTranscript } from "@/lib/demo-transcripts";
import { reportsToCallRows, type RealCall } from "@/lib/real-data";
import { useCallsStore } from "@/lib/store/calls";
import { useHealth } from "@/hooks/use-health";
import type { CallRow } from "@/lib/mock-data";

export interface DemoReportsResult {
  /** CallRow-shaped rows for the calls that finished analyzing successfully. */
  calls: CallRow[];
  /** Raw (transcript, report) pairs, for consumers that need the full report (e.g. hot topics). */
  reports: RealCall[];
  isLoading: boolean;
  /** True once every call has either succeeded or failed (no more in-flight requests). */
  isFetched: boolean;
  /** True if every call failed (nothing to show at all). */
  isError: boolean;
  /** Number of calls that failed to analyze (partial failure is still useful to show). */
  errorCount: number;
  /** True when dashboard is driven by stored/live calls (not canned demos). */
  usingLiveData: boolean;
}

/**
 * Dashboard data source for overview / agents / insights / analytics.
 *
 * Prefer persisted real calls (uploads / testlab / transcription) from
 * localStorage. Only fall back to canned demo transcripts when the store
 * is empty — so a pilot with real traffic is not mixed with synthetic demos.
 *
 * Force demos with ``NEXT_PUBLIC_FORCE_DEMO_DATA=1|true|yes`` (local UX demos).
 */
function forceDemoData(): boolean {
  const flag = process.env.NEXT_PUBLIC_FORCE_DEMO_DATA?.trim().toLowerCase();
  return flag === "1" || flag === "true" || flag === "yes";
}

export function useDemoReports(): DemoReportsResult {
  const realCalls = useCallsStore((state) => state.realCalls);
  const health = useHealth();
  const forceDemo = typeof process !== "undefined" && forceDemoData();
  const usingLiveData = !forceDemo && realCalls.length > 0;
  const runDemos = forceDemo || realCalls.length === 0;

  const queries = useQueries({
    queries: runDemos
      ? DEMO_TRANSCRIPTS.map((transcript: DemoTranscript) => ({
          queryKey: ["analyze_pipeline", "demo", transcript.id],
          queryFn: () =>
            apiClient.analyzePipeline<PipelineReport>(transcript.segments, {
              profile: "callcenter",
            }),
          staleTime: 5 * 60_000,
          retry: 1,
        }))
      : [],
  });

  const apiDown =
    health.data?.status === "offline" || health.data?.status === "unauthorized";
  const isLoading = usingLiveData ? false : queries.some((q) => q.isLoading);
  const isFetched = usingLiveData ? true : queries.every((q) => q.isFetched);
  const errorCount = usingLiveData
    ? apiDown
      ? 1
      : 0
    : queries.filter((q) => q.isError).length;
  const isError = usingLiveData
    ? Boolean(apiDown)
    : queries.length > 0 && errorCount === queries.length;

  const demoReports: RealCall[] = runDemos
    ? queries
        .map((q, i) => (q.data ? { transcript: DEMO_TRANSCRIPTS[i], report: q.data } : null))
        .filter((c): c is RealCall => c !== null)
    : [];

  const allReports: RealCall[] = usingLiveData ? realCalls : demoReports;

  return {
    calls: reportsToCallRows(allReports),
    reports: allReports,
    isLoading,
    isFetched,
    isError,
    errorCount,
    usingLiveData,
  };
}
