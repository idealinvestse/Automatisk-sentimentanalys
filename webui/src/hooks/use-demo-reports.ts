"use client";

import { useQueries } from "@tanstack/react-query";

import { apiClient, type PipelineReport } from "@/lib/api/client";
import { DEMO_TRANSCRIPTS, type DemoTranscript } from "@/lib/demo-transcripts";
import { reportsToCallRows, type RealCall } from "@/lib/real-data";
import { useCallsStore } from "@/lib/store/calls";
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
}

/**
 * Runs the real backend pipeline (`POST /analyze_pipeline`) against the
 * canned demo transcripts and maps the results to `CallRow`s. This replaces
 * the static `MOCK_CALLS` array with numbers computed by the actual
 * sentiment/intent/QA analyzers (see docs/WEBUI_MODERNIZATION_PLAN.md §6).
 *
 * Each call is cached individually by React Query, so navigating between
 * `/`, `/analytics`, `/agents` and `/insights` re-uses the same in-flight or
 * cached analysis instead of re-running the (expensive) pipeline per page.
 *
 * Merges demo calls with real calls stored in localStorage (from user uploads).
 */
export function useDemoReports(): DemoReportsResult {
  const realCalls = useCallsStore((state) => state.realCalls);

  const queries = useQueries({
    queries: DEMO_TRANSCRIPTS.map((transcript: DemoTranscript) => ({
      queryKey: ["analyze_pipeline", "demo", transcript.id],
      queryFn: () => apiClient.analyzePipeline<PipelineReport>(transcript.segments, { profile: "callcenter" }),
      staleTime: 5 * 60_000,
      retry: 1,
    })),
  });

  const isLoading = queries.some((q) => q.isLoading);
  const isFetched = queries.every((q) => q.isFetched);
  const errorCount = queries.filter((q) => q.isError).length;
  const isError = queries.length > 0 && errorCount === queries.length;

  const demoReports: RealCall[] = queries
    .map((q, i) => (q.data ? { transcript: DEMO_TRANSCRIPTS[i], report: q.data } : null))
    .filter((c): c is RealCall => c !== null);

  // Merge demo calls with real calls (real calls first)
  const allReports: RealCall[] = [...realCalls, ...demoReports];

  return {
    calls: reportsToCallRows(allReports),
    reports: allReports,
    isLoading,
    isFetched,
    isError,
    errorCount,
  };
}
