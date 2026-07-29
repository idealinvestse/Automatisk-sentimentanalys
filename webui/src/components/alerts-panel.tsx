"use client";

import { useState } from "react";
import {
  Bell,
  BellOff,
  Shield,
  ShieldAlert,
  Zap,
  WifiOff,
  RotateCcw,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { useAlerts, type AlertSeverityFilter } from "@/hooks/use-alerts";
import { useAlertingStatus, useResetCircuitBreaker } from "@/hooks/use-alerting-status";
import type { AlertItem, AlertSeverity } from "@/lib/real-data";

const SEVERITY_VARIANT: Record<AlertSeverity, "destructive" | "warning" | "secondary" | "outline"> = {
  critical: "destructive",
  high: "destructive",
  medium: "warning",
  low: "secondary",
  info: "outline",
};

const SEVERITY_LABEL: Record<AlertSeverity, string> = {
  critical: "Kritisk",
  high: "Hög",
  medium: "Medel",
  low: "Låg",
  info: "Info",
};

const FILTER_OPTIONS: { value: AlertSeverityFilter; label: string }[] = [
  { value: "all", label: "Alla" },
  { value: "critical", label: "Kritisk" },
  { value: "high", label: "Hög" },
  { value: "medium", label: "Medel" },
  { value: "low", label: "Låg" },
];

function AlertRow({ alert }: { alert: AlertItem }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <Badge variant={SEVERITY_VARIANT[alert.severity]} className="shrink-0">
            {SEVERITY_LABEL[alert.severity]}
          </Badge>
          <span className="text-sm font-medium">{alert.rule_id}</span>
        </div>
        {alert.callId && (
          <span className="shrink-0 text-xs text-muted-foreground">
            {alert.callId} · {alert.agent}
          </span>
        )}
      </div>
      <p className="text-sm text-muted-foreground">{alert.message}</p>
      {alert.recommended_actions && alert.recommended_actions.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-muted-foreground">Rekommenderade åtgärder</span>
          <ul className="flex flex-col gap-0.5">
            {alert.recommended_actions.map((action, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs">
                <Zap className="mt-0.5 size-3 shrink-0 text-warning" />
                <span>{action}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {alert.evidence_spans && alert.evidence_spans.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {alert.evidence_spans.slice(0, 3).map((e, i) => (
            <span key={i} className="rounded bg-muted/50 px-1.5 py-0.5 text-xs text-muted-foreground">
              {e.text ? `"${e.text.slice(0, 60)}${e.text.length > 60 ? "…" : ""}"` : `${e.speaker ?? ""} @ ${e.start ?? 0}s`}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function WebhookStatus() {
  const { data, isLoading, isError } = useAlertingStatus();
  const reset = useResetCircuitBreaker();

  if (isLoading) {
    return <Skeleton className="h-12 w-full" />;
  }

  if (isError || !data) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border p-3 text-sm text-muted-foreground">
        <WifiOff className="size-4 shrink-0" />
        <span>Kunde inte hämta webhook-status</span>
      </div>
    );
  }

  const webhook = data.webhook;
  const breakerOpen = webhook?.circuit_breaker_open ?? false;
  const failures = webhook?.consecutive_failures ?? 0;
  const threshold = webhook?.circuit_breaker_threshold ?? 5;

  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-border p-3">
      <div className="flex items-center gap-2">
        {breakerOpen ? (
          <ShieldAlert className="size-4 shrink-0 text-destructive" />
        ) : (
          <Shield className="size-4 shrink-0 text-success" />
        )}
        <div className="flex flex-col">
          <span className="text-sm font-medium">
            {breakerOpen ? "Circuit breaker öppen" : "Webhook aktiv"}
          </span>
          <span className="text-xs text-muted-foreground">
            {failures}/{threshold} misslyckade leveranser
          </span>
        </div>
      </div>
      {breakerOpen && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => reset.mutate()}
          disabled={reset.isPending}
          aria-label="Återställ circuit breaker"
        >
          <RotateCcw className={`size-3.5 ${reset.isPending ? "animate-spin" : ""}`} />
          Återställ
        </Button>
      )}
    </div>
  );
}

export function AlertsPanel() {
  const [filter, setFilter] = useState<AlertSeverityFilter>("all");
  const { alerts, totalCount, countsBySeverity, isLoading, isError, source } = useAlerts(filter);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-col gap-1">
            <CardTitle className="flex items-center gap-2">
              <Bell className="size-4" />
              Aktiva larm
            </CardTitle>
            <CardDescription>
              {source === "api"
                ? "Fas 4 POST /alerts (server), med fallback till pipeline-inbäddade larm."
                : "Strukturerade alerts från pipeline-resultaten (fallback när /alerts saknas)."}
            </CardDescription>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Badge variant="outline">{source === "api" ? "API" : "Pipeline"}</Badge>
            {totalCount > 0 && (
              <Badge variant={countsBySeverity.critical > 0 ? "destructive" : "warning"}>
                {totalCount} {totalCount === 1 ? "larm" : "larm"}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <WebhookStatus />

        {isLoading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
        ) : isError ? (
          <EmptyState
            icon={WifiOff}
            title="Kunde inte hämta larm"
            hint="Kontrollera att backend-API:et körs och är nåbart från webui."
          />
        ) : alerts.length === 0 ? (
          <EmptyState
            icon={BellOff}
            title="Inga aktiva larm"
            hint="Inga alerts triggades av pipelinen på demo-samtalen. Larm genereras vid hög churn-risk, eskalering, compliance-problem m.m."
          />
        ) : (
          <>
            <div className="flex flex-wrap gap-1.5">
              {FILTER_OPTIONS.map((opt) => {
                const count = opt.value === "all" ? totalCount : countsBySeverity[opt.value] ?? 0;
                if (opt.value !== "all" && count === 0) return null;
                return (
                  <button
                    key={opt.value}
                    onClick={() => setFilter(opt.value)}
                    className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${
                      filter === opt.value
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-border text-muted-foreground hover:bg-muted"
                    }`}
                  >
                    {opt.label} ({count})
                  </button>
                );
              })}
            </div>
            <div className="flex flex-col gap-2">
              {alerts.map((alert, i) => (
                <AlertRow key={`${alert.callId}-${alert.rule_id}-${i}`} alert={alert} />
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
