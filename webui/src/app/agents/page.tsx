"use client";

import { Users, AlertTriangle, WifiOff } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { useAgentPerformance } from "@/hooks/use-agent-performance";

export default function AgentsPage() {
  const { calls, reports, isLoading: reportsLoading, isError, usingLiveData } = useDemoReports();
  const { rows: agents, isLoading: agentsLoading } = useAgentPerformance(calls, reports);
  const isLoading = reportsLoading || (calls.length > 0 && agentsLoading && agents.length === 0);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Agentprestanda</h1>
        <p className="text-sm text-muted-foreground">
          Sentiment, QA-poäng och larm per agent, aggregerat via{" "}
          <code>POST /agent_performance/&#123;agent_id&#125;</code> över{" "}
          {usingLiveData ? "sparade samtal." : "demosamtalen."}
        </p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-36 w-full" />
          ))}
        </div>
      ) : isError ? (
        <EmptyState
          icon={WifiOff}
          title="Kunde inte hämta agentdata"
          hint="Kontrollera att backend-API:et körs och är nåbart från webui."
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {agents.map((agent) => (
            <Card key={agent.agent}>
              <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
                <div className="flex items-center gap-2">
                  <div className="flex size-9 items-center justify-center rounded-full bg-primary/10 text-primary">
                    <Users className="size-4" />
                  </div>
                  <div className="flex flex-col">
                    <CardTitle className="text-sm font-semibold text-foreground">
                      {agent.agent}
                    </CardTitle>
                    <CardDescription>{agent.calls} samtal</CardDescription>
                  </div>
                </div>
                <div className="flex items-center gap-1.5">
                  <Badge variant={agent.source === "api" ? "outline" : "warning"}>
                    {agent.source === "api" ? "API" : "Lokal fallback"}
                  </Badge>
                  {agent.alertCount > 0 ? (
                    <Badge variant="warning" className="gap-1">
                      <AlertTriangle className="size-3" />
                      {agent.alertCount}
                    </Badge>
                  ) : null}
                </div>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs text-muted-foreground">Snitt-sentiment</span>
                  <span className="text-lg font-semibold">
                    {Math.round(agent.avgSentiment * 100)}%
                  </span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs text-muted-foreground">QA-poäng</span>
                  <span
                    className={`text-lg font-semibold ${
                      agent.avgQaScore >= 75 ? "text-success" : "text-warning-text"
                    }`}
                  >
                    {Math.round(agent.avgQaScore)}/100
                  </span>
                </div>
                {agent.empathyScore !== null ? (
                  <div className="col-span-2 flex flex-col gap-0.5 border-t border-border pt-2">
                    <span className="text-xs text-muted-foreground">
                      Empati (Fas 4-aggregat)
                    </span>
                    <span className="text-sm font-medium">
                      {Math.round(agent.empathyScore * 100)}%
                    </span>
                  </div>
                ) : null}
                {agent.complianceFlagCount > 0 ? (
                  <div className="col-span-2 flex items-center gap-1.5 border-t border-border pt-2">
                    <Badge variant="destructive" className="text-[10px]">
                      {agent.complianceFlagCount} compliance-flagg
                      {agent.complianceFlagCount === 1 ? "" : "or"}
                    </Badge>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
