"use client";

import { TrendingUp, TrendingDown, Minus, Flame, WifiOff, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { AlertsPanel } from "@/components/alerts-panel";
import { QaScorePanel } from "@/components/qa-score-panel";
import { SemanticSearchPanel } from "@/components/semantic-search-panel";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { useHotTopics } from "@/hooks/use-hot-topics";
import type { HotTopicItem } from "@/lib/api/client";

const TREND_ICON: Record<HotTopicItem["trend"], typeof TrendingUp> = {
  up: TrendingUp,
  down: TrendingDown,
  stable: Minus,
};

const TREND_LABEL: Record<HotTopicItem["trend"], string> = {
  up: "Ökande",
  down: "Minskande",
  stable: "Stabil",
};

const TREND_TONE: Record<HotTopicItem["trend"], "destructive" | "success" | "secondary"> = {
  up: "destructive",
  down: "success",
  stable: "secondary",
};

export default function InsightsPage() {
  const { reports, isLoading: reportsLoading, isError: reportsError } = useDemoReports();
  const hotTopics = useHotTopics(reports);

  const isLoading = reportsLoading || hotTopics.isLoading;
  const isError = reportsError || hotTopics.isError;
  const topics = [...(hotTopics.data?.hot_topics ?? [])].sort((a, b) => b.volume - a.volume);
  const maxVolume = Math.max(1, ...topics.map((t) => t.volume));

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Fas 4 Insikter</h1>
        <p className="text-sm text-muted-foreground">
          Hot topics, semantisk sökning, QA-score och larm mot Fas 4-API:erna över aktuella samtal.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Hot topics</CardTitle>
          <CardDescription>Rankade efter samtalsvolym.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {isLoading ? (
            Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-16 w-full" />)
          ) : isError ? (
            <EmptyState
              icon={WifiOff}
              title="Kunde inte hämta hot topics"
              hint="Kontrollera att backend-API:et körs och är nåbart från webui."
            />
          ) : topics.length === 0 ? (
            <EmptyState
              icon={Sparkles}
              title="Inga hot topics hittades"
              hint="Aggregatorn behöver fler/mer varierade samtal för att identifiera återkommande ämnen."
            />
          ) : (
            topics.map((topic) => {
              const TrendIcon = TREND_ICON[topic.trend];
              const positivityPct = Math.round(((topic.avg_sentiment + 1) / 2) * 100);
              return (
                <div
                  key={topic.topic}
                  className="flex flex-col gap-2 rounded-lg border border-border p-3 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <Flame className="size-4" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium capitalize">{topic.topic}</span>
                      <span className="text-xs text-muted-foreground">
                        {topic.volume} samtal · {positivityPct}% snitt-sentiment
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 sm:w-64">
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${(topic.volume / maxVolume) * 100}%` }}
                      />
                    </div>
                    <Badge variant={TREND_TONE[topic.trend]} className="shrink-0 gap-1">
                      <TrendIcon className="size-3" />
                      {TREND_LABEL[topic.trend]}
                    </Badge>
                  </div>
                </div>
              );
            })
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <SemanticSearchPanel reports={reports} />
        <QaScorePanel reports={reports} />
      </div>

      <AlertsPanel />
    </div>
  );
}
