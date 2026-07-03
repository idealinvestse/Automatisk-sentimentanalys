"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { apiClient, type AlertingStatusResponse } from "@/lib/api/client";

/**
 * Polls `GET /alerting/status` for webhook + circuit breaker health.
 * Used by the alerts panel to show whether the webhook delivery pipeline
 * is healthy or if the circuit breaker has tripped.
 */
export function useAlertingStatus() {
  return useQuery({
    queryKey: ["alerting", "status"],
    queryFn: () => apiClient.getAlertingStatus<AlertingStatusResponse>(),
    staleTime: 30_000,
    retry: 1,
  });
}

/**
 * Manually reset the webhook circuit breaker via
 * `POST /alerting/reset-circuit-breaker`. Invalidates the status query
 * on success so the UI reflects the reset immediately.
 */
export function useResetCircuitBreaker() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => apiClient.resetCircuitBreaker<AlertingStatusResponse>(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerting", "status"] });
    },
  });
}
