"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Bell, BellOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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

interface CallAlertsSectionProps {
  alerts: AlertItem[];
}

/** Compact per-call alerts section for the call detail page. */
export function CallAlertsSection({ alerts }: CallAlertsSectionProps) {
  const [expanded, setExpanded] = useState(true);

  if (alerts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BellOff className="size-4" />
            Larm för detta samtal
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Inga alerts triggades för detta samtal.
          </p>
        </CardContent>
      </Card>
    );
  }

  const shown = alerts.slice(0, 15);

  return (
    <Card>
      <CardHeader>
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-2 text-left"
          aria-expanded={expanded}
          aria-label="Växla larmsektion"
        >
          {expanded ? (
            <ChevronDown className="size-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-4 text-muted-foreground" />
          )}
          <Bell className="size-4" />
          <CardTitle>Larm för detta samtal</CardTitle>
          <Badge variant={alerts.some((a) => a.severity === "critical") ? "destructive" : "warning"}>
            {alerts.length}
          </Badge>
        </button>
      </CardHeader>
      {expanded && (
        <CardContent className="flex flex-col gap-2">
          {shown.map((alert, i) => (
            <div key={i} className="flex flex-col gap-1.5 rounded-lg border border-border p-2.5">
              <div className="flex items-center gap-2">
                <Badge variant={SEVERITY_VARIANT[alert.severity]} className="shrink-0">
                  {SEVERITY_LABEL[alert.severity]}
                </Badge>
                <span className="text-sm font-medium">{alert.rule_id}</span>
              </div>
              <p className="text-sm text-muted-foreground">{alert.message}</p>
              {alert.evidence_spans && alert.evidence_spans.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {alert.evidence_spans.slice(0, 3).map((e, j) => (
                    <span
                      key={j}
                      className="rounded bg-muted/50 px-1.5 py-0.5 text-xs text-muted-foreground"
                    >
                      {e.text
                        ? `"${e.text.slice(0, 60)}${e.text.length > 60 ? "…" : ""}"`
                        : `${e.speaker ?? ""} @ ${e.start ?? 0}s`}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {alerts.length > 15 && (
            <span className="text-xs text-muted-foreground">
              Visar 15 av {alerts.length} larm.
            </span>
          )}
        </CardContent>
      )}
    </Card>
  );
}
