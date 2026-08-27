import { create } from "zustand";
import { persist } from "zustand/middleware";

import { apiClient } from "@/lib/api/client";
import { notifyApiError } from "@/lib/notify";
import type { RealCall } from "@/lib/real-data";

interface CallsState {
  /** Real transcribed calls (local cache; synced to server when API is reachable) */
  realCalls: RealCall[];
  /** Add a new real call to the registry */
  addRealCall: (call: RealCall) => void;
  /** Remove a call from the registry */
  removeRealCall: (id: string) => void;
  /** Clear all real calls */
  clearRealCalls: () => void;
  /** Pull server-side call history into the local cache (best-effort). */
  syncFromServer: () => Promise<void>;
}

function persistToServer(call: RealCall): void {
  const id = call.transcript.id;
  void apiClient
    .saveCall(id, {
      transcript: call.transcript as unknown as Record<string, unknown>,
      report: call.report as unknown as Record<string, unknown>,
      meta: { source: "webui" },
    })
    .catch((err) => {
      notifyApiError(err, "Kunde inte spara samtalet på servern");
    });
}

export const useCallsStore = create<CallsState>()(
  persist(
    (set, get) => ({
      realCalls: [],
      addRealCall: (call) => {
        set((state) => {
          const exists = state.realCalls.some((c) => c.transcript.id === call.transcript.id);
          if (exists) return state;
          return { realCalls: [call, ...state.realCalls] };
        });
        persistToServer(call);
      },
      removeRealCall: (id) => {
        set((state) => ({
          realCalls: state.realCalls.filter((c) => c.transcript.id !== id),
        }));
        void apiClient.deleteCall(id).catch((err) => {
          notifyApiError(err, "Kunde inte ta bort samtalet på servern");
        });
      },
      clearRealCalls: () => set({ realCalls: [] }),
      syncFromServer: async () => {
        try {
          const res = await apiClient.listCalls<{
            calls: Array<{
              id: string;
              transcript?: RealCall["transcript"];
              report?: RealCall["report"];
            }>;
          }>(100);
          const remote: RealCall[] = (res.calls ?? [])
            .filter((c) => c.transcript && c.report)
            .map((c) => ({
              transcript: c.transcript as RealCall["transcript"],
              report: c.report as RealCall["report"],
            }));
          if (remote.length === 0) return;
          const local = get().realCalls;
          const byId = new Map<string, RealCall>();
          for (const call of [...remote, ...local]) {
            byId.set(call.transcript.id, call);
          }
          set({ realCalls: [...byId.values()] });
        } catch {
          /* ignore — keep local cache */
        }
      },
    }),
    {
      name: "sentiment-calls-storage",
      version: 2,
    },
  ),
);
