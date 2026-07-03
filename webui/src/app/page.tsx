"use client";

import { useRouter } from "next/navigation";
import { PhoneCall, Smile, ShieldCheck, AlertTriangle, WifiOff } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { KpiCard } from "@/components/kpi-card";
import { CallsTable } from "@/components/calls-table";
import { useHealth } from "@/hooks/use-health";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { summarizeKpis } from "@/lib/mock-data";

export default function OverviewPage() {
  const router = useRouter();
  const { data: connected } = useHealth();
  const { calls, isLoading, isError, errorCount } = useDemoReports();
  const kpis = summarizeKpis(calls);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Översikt</h1>
          <p className="text-sm text-muted-foreground">
            Senaste samtal, KPI:er och status för call center-analysen.
          </p>
        </div>
        <Badge variant={connected ? "success" : "warning"}>
          {connected ? "Backend ansluten" : "Backend ej tillgänglig"}
        </Badge>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      ) : isError ? (
        <EmptyState
          icon={WifiOff}
          title="Kunde inte analysera demosamtalen"
          hint="Kontrollera att backend-API:et (uvicorn src.api:app) körs och är nåbart från webui."
        />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <KpiCard label="Samtal idag" value={String(kpis.totalCalls)} icon={PhoneCall} hint="Senaste 24h (demo)" />
            <KpiCard
              label="Snitt-sentiment"
              value={`${Math.round(kpis.avgSentiment * 100)}%`}
              icon={Smile}
              tone={kpis.avgSentiment >= 0.5 ? "success" : "warning"}
              hint="Alla kanaler"
            />
            <KpiCard
              label="QA-poäng"
              value={`${Math.round(kpis.avgQaScore)}/100`}
              icon={ShieldCheck}
              tone={kpis.avgQaScore >= 75 ? "success" : "warning"}
              hint="Genomsnitt"
            />
            <KpiCard
              label="Aktiva larm"
              value={String(kpis.activeAlerts)}
              icon={AlertTriangle}
              tone={kpis.activeAlerts > 0 ? "warning" : "default"}
              hint="Kräver åtgärd"
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Senaste samtal</CardTitle>
            </CardHeader>
            <CardContent>
              <CallsTable data={calls} onSelectCall={(callId) => router.push(`/calls/${callId}`)} />
            </CardContent>
          </Card>
        </>
      )}

      {errorCount > 0 && !isError ? (
        <p className="text-xs text-warning-text">
          {errorCount} av demosamtalen kunde inte analyseras och visas inte ovan.
        </p>
      ) : null}

      <p className="text-xs text-muted-foreground">
        Samtalen ovan är kanoniska svenska demo-transkript (se{" "}
        <code>src/lib/demo-transcripts.ts</code>), men sentiment, QA-poäng och risknivå beräknas av
        den riktiga backend-pipelinen via <code>POST /analyze_pipeline</code> — inga hårdkodade
        siffror. Se docs/WEBUI_MODERNIZATION_PLAN.md §6.
      </p>
    </div>
  );
}
