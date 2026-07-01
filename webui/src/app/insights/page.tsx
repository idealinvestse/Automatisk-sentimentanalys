"use client";

import { TrendingUp, TrendingDown, Minus, Flame } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { MOCK_HOT_TOPICS, type HotTopic } from "@/lib/mock-data";

const TREND_ICON: Record<HotTopic["trend"], typeof TrendingUp> = {
  up: TrendingUp,
  down: TrendingDown,
  flat: Minus,
};

const TREND_LABEL: Record<HotTopic["trend"], string> = {
  up: "Ökande",
  down: "Minskande",
  flat: "Stabil",
};

const TREND_TONE: Record<HotTopic["trend"], "destructive" | "success" | "secondary"> = {
  up: "destructive",
  down: "success",
  flat: "secondary",
};

export default function InsightsPage() {
  const topics = [...MOCK_HOT_TOPICS].sort((a, b) => b.mentions - a.mentions);
  const maxMentions = Math.max(...topics.map((t) => t.mentions));

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Fas 4 Insikter</h1>
        <p className="text-sm text-muted-foreground">
          Hot topics från samtalsvolymen den senaste veckan (demo-data).
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Hot topics</CardTitle>
          <CardDescription>Rankade efter antal omnämnanden, senaste 7 dagarna.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {topics.map((topic) => {
            const TrendIcon = TREND_ICON[topic.trend];
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
                    <span className="text-sm font-medium">{topic.topic}</span>
                    <span className="text-xs text-muted-foreground">
                      {topic.mentions} omnämnanden · {Math.round(topic.avgSentiment * 100)}% snitt-sentiment
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-3 sm:w-64">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${(topic.mentions / maxMentions) * 100}%` }}
                    />
                  </div>
                  <Badge variant={TREND_TONE[topic.trend]} className="shrink-0 gap-1">
                    <TrendIcon className="size-3" />
                    {TREND_LABEL[topic.trend]}
                  </Badge>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
