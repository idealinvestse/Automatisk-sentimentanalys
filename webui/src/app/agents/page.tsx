"use client";

import { Users, AlertTriangle } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MOCK_CALLS, summarizeAgents } from "@/lib/mock-data";

export default function AgentsPage() {
  const agents = summarizeAgents(MOCK_CALLS);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Agentprestanda</h1>
        <p className="text-sm text-muted-foreground">
          Sentiment, QA-poäng och larm per agent (demo-data).
        </p>
      </div>

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
              {agent.alertCount > 0 ? (
                <Badge variant="warning" className="gap-1">
                  <AlertTriangle className="size-3" />
                  {agent.alertCount}
                </Badge>
              ) : null}
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
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
