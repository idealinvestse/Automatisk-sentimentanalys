"use client";

import { useRouter } from "next/navigation";
import { PhoneCall, Smile, ShieldCheck, AlertTriangle, WifiOff, Gauge, Lightbulb, ArrowUpCircle, CheckCircle2 } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { KpiCard } from "@/components/kpi-card";
import { CallsTable } from "@/components/calls-table";
import { isApiConnected, useHealth } from "@/hooks/use-health";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { summarizeKpis } from "@/lib/mock-data";
import { extractCustomerEffort, extractCoaching, extractUpsell, extractResolutionProbability } from "@/lib/real-data";

export default function OverviewPage() {
  const router = useRouter();
  const { data: health } = useHealth();
  const connected = isApiConnected(health);
  const { calls, reports, isLoading, isError, errorCount } = useDemoReports();
  const kpis = summarizeKpis(calls);

  // Fas 5: aggregate analyzer KPIs across all demo reports
  const cesScores = reports
    .map((r) => extractCustomerEffort(r.report)?.overall_ces)
    .filter((v): v is number => typeof v === "number");
  const avgCes = cesScores.length > 0 ? cesScores.reduce((s, v) => s + v, 0) / cesScores.length : null;

  const coachingCounts = reports
    .map((r) => extractCoaching(r.report)?.insight_count ?? 0);
  const totalCoachingInsights = coachingCounts.reduce((s, v) => s + v, 0);

  const upsellCounts = reports
    .map((r) => extractUpsell(r.report)?.count ?? 0);
  const totalUpsellOpps = upsellCounts.reduce((s, v) => s + v, 0);

  const resolutionProbs = reports
    .map((r) => extractResolutionProbability(r.report)?.resolution_probability)
    .filter((v): v is number => typeof v === "number");
  const avgResolution = resolutionProbs.length > 0
    ? resolutionProbs.reduce((s, v) => s + v, 0) / resolutionProbs.length
    : null;

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

          {/* Fas 5: Analyzer aggregate KPIs */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <KpiCard
              label="Snitt CES"
              value={avgCes !== null ? `${Math.round(avgCes)}/100` : "—"}
              icon={Gauge}
              tone={avgCes !== null ? (avgCes < 40 ? "success" : "warning") : "default"}
              hint="Kundinsats (lägre = bättre)"
            />
            <KpiCard
              label="Coaching-insikter"
              value={String(totalCoachingInsights)}
              icon={Lightbulb}
              tone={totalCoachingInsights > 0 ? "warning" : "default"}
              hint="Prioriterade rekommendationer"
            />
            <KpiCard
              label="Upsell-möjligheter"
              value={String(totalUpsellOpps)}
              icon={ArrowUpCircle}
              tone={totalUpsellOpps > 0 ? "success" : "default"}
              hint="Identifierade tillfällen"
            />
            <KpiCard
              label="Snitt lösningsgrad"
              value={avgResolution !== null ? `${Math.round(avgResolution)}%` : "—"}
              icon={CheckCircle2}
              tone={avgResolution !== null ? (avgResolution > 60 ? "success" : "warning") : "default"}
              hint="Sannolikhet för resolution"
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
        siffror.
      </p>
    </div>
  );
}
