"use client";

import { PhoneCall, Smile, ShieldCheck, AlertTriangle } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { KpiCard } from "@/components/kpi-card";
import { useHealth } from "@/hooks/use-health";

export default function OverviewPage() {
  const { data: connected } = useHealth();

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
          {connected ? "Data från API" : "Demo-data (API ej tillgänglig)"}
        </Badge>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard label="Samtal idag" value="—" icon={PhoneCall} hint="Senaste 24h" />
        <KpiCard label="Snitt-sentiment" value="—" icon={Smile} tone="success" hint="Alla kanaler" />
        <KpiCard label="QA-poäng" value="—" icon={ShieldCheck} tone="default" hint="Genomsnitt" />
        <KpiCard label="Aktiva larm" value="—" icon={AlertTriangle} tone="warning" hint="Kräver åtgärd" />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Senaste samtal</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Samtalstabell migreras i Fas 1 (se docs/WEBUI_MODERNIZATION_PLAN.md). Denna sida
          demonstrerar det nya designsystemet: kort, badges och typografi-skala kopplat mot
          samma FastAPI-backend (<code className="text-xs">/health</code>) som den befintliga
          NiceGUI-dashboarden.
        </CardContent>
      </Card>
    </div>
  );
}
