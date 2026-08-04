"use client";

import { useEffect } from "react";

import { useCallsStore } from "@/lib/store/calls";

/** Best-effort pull of server-persisted calls into the local cache on mount. */
export function CallSync() {
  const syncFromServer = useCallsStore((s) => s.syncFromServer);
  useEffect(() => {
    void syncFromServer();
  }, [syncFromServer]);
  return null;
}
