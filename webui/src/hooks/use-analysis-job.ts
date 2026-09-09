"use client";

import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { apiClient, type AnalysisJobStatus } from "@/lib/api/client";

const TERMINAL_STATES = new Set([
  "completed",
  "completed_degraded",
  "failed",
  "cancelled",
  "interrupted",
]);

export function useAnalysisJob() {
  const [jobId, setJobId] = React.useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem("active-analysis-job");
  });

  const status = useQuery<AnalysisJobStatus>({
    queryKey: ["analysis-job", jobId],
    queryFn: () => apiClient.getAnalysisJob(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const state = query.state.data?.status;
      return state && TERMINAL_STATES.has(state) ? false : 2_000;
    },
  });

  const start = useMutation({
    mutationFn: (segments: unknown[]) =>
      apiClient.createAnalysisJob(
        segments,
        { device: "cpu", use_mistral_llm: true, deep_analysis: true },
        crypto.randomUUID(),
      ),
    onSuccess: (job) => {
      setJobId(job.job_id);
      window.localStorage.setItem("active-analysis-job", job.job_id);
    },
  });

  const cancel = useMutation({
    mutationFn: () => apiClient.cancelAnalysisJob(jobId!),
    onSuccess: () => status.refetch(),
  });

  const clear = React.useCallback(() => {
    setJobId(null);
    window.localStorage.removeItem("active-analysis-job");
  }, []);

  return { jobId, status, start, cancel, clear };
}
