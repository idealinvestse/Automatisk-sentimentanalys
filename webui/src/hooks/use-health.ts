"use client";

import { useQuery } from "@tanstack/react-query";

import { apiClient } from "@/lib/api/client";

/** Poll backend /health every 15s to drive the connection badge. */
export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiClient.health(),
    refetchInterval: 15_000,
  });
}
