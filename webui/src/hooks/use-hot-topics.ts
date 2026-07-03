"use client";

import { useQuery } from "@tanstack/react-query";

import { apiClient, type HotTopicsResponse } from "@/lib/api/client";
import type { RealCall } from "@/lib/real-data";

/**
 * Fetches real aggregated hot topics via `POST /insights/hot_topics` (Fas 4)
 * over all successfully analyzed demo calls. `avg_sentiment` from the API is
 * on a -1..1 scale (see src/llm/schemas.py::HotTopic); callers should
 * normalize with `(avg_sentiment + 1) / 2` for 0..1 display.
 */
export function useHotTopics(reports: RealCall[]) {
  const segmentsList = reports.map((r) => r.transcript.segments);

  return useQuery({
    queryKey: ["hot_topics", segmentsList.length],
    queryFn: () => apiClient.getHotTopics<HotTopicsResponse>(segmentsList),
    enabled: segmentsList.length > 0,
    staleTime: 5 * 60_000,
    retry: 1,
  });
}
