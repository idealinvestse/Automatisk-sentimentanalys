import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { RealCall } from "@/lib/real-data";

interface CallsState {
  /** Real transcribed calls stored in localStorage */
  realCalls: RealCall[];
  /** Add a new real call to the registry */
  addRealCall: (call: RealCall) => void;
  /** Remove a call from the registry */
  removeRealCall: (id: string) => void;
  /** Clear all real calls */
  clearRealCalls: () => void;
}

export const useCallsStore = create<CallsState>()(
  persist(
    (set) => ({
      realCalls: [],
      addRealCall: (call) =>
        set((state) => {
          // Avoid duplicates by ID
          const exists = state.realCalls.some((c) => c.transcript.id === call.transcript.id);
          if (exists) return state;
          return { realCalls: [call, ...state.realCalls] };
        }),
      removeRealCall: (id) =>
        set((state) => ({
          realCalls: state.realCalls.filter((c) => c.transcript.id !== id),
        })),
      clearRealCalls: () => set({ realCalls: [] }),
    }),
    {
      name: "sentiment-calls-storage",
      version: 1,
    },
  ),
);
