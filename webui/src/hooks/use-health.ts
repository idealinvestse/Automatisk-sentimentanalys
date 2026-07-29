"use client";

import { useQuery } from "@tanstack/react-query";

import { apiClient, type ApiConnectionStatus } from "@/lib/api/client";

/** Poll backend connectivity + auth every 15s to drive the connection badge. */
export function useHealth() {
  return useQuery<ApiConnectionStatus>({
    queryKey: ["health"],
    queryFn: () => apiClient.connectionStatus(),
    refetchInterval: 15_000,
  });
}

/** True when API is reachable and auth is OK (or not required). */
export function isApiConnected(status: ApiConnectionStatus | undefined): boolean {
  return status?.status === "ok";
}
