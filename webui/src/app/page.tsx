"use client";

import { PhoneCall, Smile, ShieldCheck, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { KpiCard } from "@/components/kpi-card";
import { CallsTable } from "@/components/calls-table";
import { useHealth } from "@/hooks/use-health";
import { MOCK_CALLS, summarizeKpis } from "@/lib/mock-data";

export default function OverviewPage() {
  const { data: connected } = useHealth();
  const kpis = summarizeKpis(MOCK_CALLS);

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
          <CallsTable
            data={MOCK_CALLS}
            onSelectCall={(callId) =>
              toast.info(`Samtalsdetalj för ${callId} kommer i Fas 2`)
            }
          />
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        Data ovan är demo-/mockdata (se <code>src/lib/mock-data.ts</code>) i väntan på beslut om
        datakälla, se docs/WEBUI_MODERNIZATION_PLAN.md.
      </p>
    </div>
  );
}
