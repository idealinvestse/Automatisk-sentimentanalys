"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Scale, Sparkles, AlertCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import type { LlmJudgeResult, LlmJudgeVerdict } from "@/lib/real-data";

const SENTIMENT_COLOR: Record<string, string> = {
  positive: "text-success",
  positiv: "text-success",
  negative: "text-destructive",
  negativ: "text-destructive",
  neutral: "text-muted-foreground",
};

function confidenceColor(conf: number): string {
  if (conf >= 0.7) return "text-success";
  if (conf >= 0.4) return "text-warning";
  return "text-destructive";
}

function VerdictRow({ verdict }: { verdict: LlmJudgeVerdict }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const changed = verdict.original_sentiment !== verdict.judge_label;

  return (
    <div
      className={`flex flex-col gap-1.5 rounded-lg border p-2.5 ${
        changed ? "border-warning/50 bg-warning/5" : "border-border"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">Seg #{verdict.segment_index}</span>
        {changed && (
          <Badge variant="warning" className="shrink-0 text-xs">
            Ändrad
          </Badge>
        )}
      </div>
      <div className="flex items-center gap-2 text-sm">
        <span className={SENTIMENT_COLOR[verdict.original_sentiment.toLowerCase()] ?? "text-muted-foreground"}>
          {verdict.original_sentiment}
        </span>
        <span className="text-xs text-muted-foreground">
          ({Math.round(verdict.original_confidence * 100)}%)
        </span>
        <span className="text-muted-foreground">→</span>
        <span className={SENTIMENT_COLOR[verdict.judge_label.toLowerCase()] ?? "text-muted-foreground"}>
          {verdict.judge_label}
        </span>
        <span className={`text-xs ${confidenceColor(verdict.judge_confidence)}`}>
          ({Math.round(verdict.judge_confidence * 100)}%)
        </span>
      </div>
      {verdict.reasoning && (
        <button
          onClick={() => setShowReasoning((v) => !v)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          aria-expanded={showReasoning}
        >
          {showReasoning ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
          Motivering
        </button>
      )}
      {showReasoning && verdict.reasoning && (
        <p className="text-xs text-muted-foreground">{verdict.reasoning}</p>
      )}
    </div>
  );
}

interface LlmJudgePanelProps {
  result: LlmJudgeResult | null;
}

/** LLM Judge panel for the call detail page — shows re-evaluation of
 *  low-confidence sentiment segments by an LLM judge. */
export function LlmJudgePanel({ result }: LlmJudgePanelProps) {
  const [showOnlyChanged, setShowOnlyChanged] = useState(false);

  if (!result || result.verdicts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Scale className="size-4" />
            LLM Judge
          </CardTitle>
          <CardDescription>LLM-ombedömning av lågkonfidens-segment.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Sparkles className="size-4" />
            <span>
              {result
                ? `LLM Judge kördes men bedömde inga segment (triggerades ej).`
                : "LLM Judge aktiverades inte för detta samtal."}
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const changedCount = result.verdicts.filter(
    (v) => v.original_sentiment !== v.judge_label,
  ).length;
  const filtered = showOnlyChanged
    ? result.verdicts.filter((v) => v.original_sentiment !== v.judge_label)
    : result.verdicts;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Scale className="size-4" />
          LLM Judge
        </CardTitle>
        <CardDescription>
          {result.triggered_segments} segment bedömda · {changedCount} ändrade ·{" "}
          {result.fallback_used ? "fallback-läge" : "LLM aktiv"} ·{" "}
          ${result.total_cost_usd.toFixed(4)}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {result.budget_exceeded && (
          <div className="flex items-center gap-2 rounded-lg border border-warning/50 bg-warning/5 p-2 text-xs text-warning">
            <AlertCircle className="size-3.5 shrink-0" />
            <span>Budget överskriden — vissa segment bedömdes inte.</span>
          </div>
        )}

        <button
          onClick={() => setShowOnlyChanged((v) => !v)}
          className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
          aria-pressed={showOnlyChanged}
        >
          {showOnlyChanged ? (
            <ChevronDown className="size-3" />
          ) : (
            <ChevronRight className="size-3" />
          )}
          {showOnlyChanged ? "Visa alla" : "Visa endast ändrade"} ({changedCount})
        </button>

        <div className="flex flex-col gap-2">
          {filtered.map((v, i) => (
            <VerdictRow key={i} verdict={v} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
