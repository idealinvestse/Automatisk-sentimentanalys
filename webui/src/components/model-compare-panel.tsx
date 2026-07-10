"use client";

import { GitCompareArrows } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { type PipelineCompareResponse } from "@/lib/api/client";
import { type RoutingTier, tierToModel } from "@/lib/routing-tier";

export interface ModelComparePanelProps {
  data: PipelineCompareResponse | undefined;
  isLoading: boolean;
  tiers?: RoutingTier[];
}

const DEFAULT_TIERS: RoutingTier[] = ["fast", "balanced", "deep"];

function formatCost(usd: number | null | undefined): string {
  if (usd == null) return "—";
  return `$${usd.toFixed(4)}`;
}

export function ModelComparePanel({
  data,
  isLoading,
  tiers = DEFAULT_TIERS,
}: ModelComparePanelProps) {
  const models = tiers.map(tierToModel);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <GitCompareArrows className="size-4" />
            Modelljämförelse
          </CardTitle>
          <CardDescription>Kör samma segment genom FAST / BALANCED / DEEP…</CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">Analyserar…</CardContent>
      </Card>
    );
  }

  if (!data || Object.keys(data.results).length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <GitCompareArrows className="size-4" />
            Modelljämförelse
          </CardTitle>
          <CardDescription>
            Jämför upp till tre modeller side-by-side via{" "}
            <code>/analyze_pipeline/compare</code>.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Aktivera LLM och kör jämförelse för att se resultat.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2 text-base">
              <GitCompareArrows className="size-4" />
              Modelljämförelse
            </CardTitle>
            <CardDescription>
              Totalt {data.total_processing_time_s.toFixed(1)}s
              {data.total_cost_usd != null ? ` · ${formatCost(data.total_cost_usd)}` : ""}
            </CardDescription>
          </div>
          {data.budget_exceeded ? (
            <Badge variant="warning">Budget överskriden</Badge>
          ) : (
            <Badge variant="success">Inom budget</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-[1fr_auto_auto_auto_auto] gap-x-4 gap-y-1 text-xs font-medium text-muted-foreground">
          <span>Modell</span>
          <span>Sentiment</span>
          <span>QA</span>
          <span>Kostnad</span>
          <span>Tid (s)</span>
        </div>
        {models.map((model) => {
          const row = data.results[model];
          if (!row) return null;
          return (
            <div
              key={model}
              className="grid grid-cols-[1fr_auto_auto_auto_auto] gap-x-4 gap-y-1 border-t border-border pt-2 text-sm"
            >
              <span className="truncate font-mono text-xs">{model.split("/").pop()}</span>
              <span>{row.sentiment_label ?? "—"}</span>
              <span>{row.qa_score != null ? row.qa_score.toFixed(2) : "—"}</span>
              <span>{formatCost(row.llm_cost_usd)}</span>
              <span>{row.processing_time_s.toFixed(2)}</span>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
