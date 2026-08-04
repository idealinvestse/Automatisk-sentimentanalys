"use client";

/**
 * Surfaces pipeline `mode: "degraded"` / `degraded: string[]` from a report.
 */
export function DegradedBanner({
  mode,
  degraded,
}: {
  mode?: string | null;
  degraded?: string[] | null;
}) {
  const reasons = (degraded ?? []).filter(Boolean);
  if (mode !== "degraded" && reasons.length === 0) {
    return null;
  }
  return (
    <div
      role="status"
      className="mb-4 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-950 dark:text-amber-100"
    >
      <p className="font-medium">Reducerad analys (graceful degradation)</p>
      <p className="mt-1 opacity-90">
        Vissa steg hoppades över eller föll tillbaka till lokal/heuristisk väg.
        {reasons.length > 0 ? ` Orsaker: ${reasons.join(", ")}.` : null}
      </p>
    </div>
  );
}
