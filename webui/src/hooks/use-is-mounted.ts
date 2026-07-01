"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

/**
 * True only once the component has hydrated on the client. Used to defer
 * rendering of client-only state (e.g. next-themes' resolved theme) so the
 * server-rendered markup matches the first client render and avoids a
 * hydration mismatch, without calling setState from inside an effect.
 */
export function useIsMounted(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
