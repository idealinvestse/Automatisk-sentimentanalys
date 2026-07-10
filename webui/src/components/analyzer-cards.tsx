"use client";

import * as React from "react";
import {
  Heart,
  TrendingDown,
  TrendingUp,
  Minus,
  Target,
  Lightbulb,
  Gauge,
  Ear,
  Route,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ArrowUpCircle,
  MessageSquare,
  Users,
  Shield,
  GitBranch,
  History,
  Sparkles,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type {
  EmotionSegmentResult,
  AspectItem,
  DerivedCallSentiment,
  TrajectoryResult,
  RootCauseResult,
  CoachingResult,
  CustomerEffortResult,
  ActiveListeningResult,
  EmpathyResult,
  ResolutionProbabilityResult,
  MultiTurnJourneyResult,
  UpsellResult,
  ComplianceRiskResult,
  RoleClassifierResult,
  PredictiveResult,
  DeepPathCCP,
  DegradationInfo,
  AnalyzerRouting,
  OverrideProvenanceEntry,
  PipelineReport,
} from "@/lib/api/client";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const EMOTION_LABELS_SV: Record<string, string> = {
  frustration: "Frustration",
  ilska: "Ilska",
  besvikelse: "Besvikelse",
  förvirring: "Förvirring",
  tillfredsställelse: "Tillfredsställelse",
  neutral: "Neutral",
  oro: "Oro",
  glädje: "Glädje",
};

function scoreColor(score: number, thresholds: [number, number] = [50, 70]): string {
  if (score >= thresholds[1]) return "text-success";
  if (score >= thresholds[0]) return "text-warning";
  return "text-destructive";
}

function riskBadgeVariant(level: string): "success" | "warning" | "destructive" {
  if (level === "high" || level === "critical") return "destructive";
  if (level === "medium") return "warning";
  return "success";
}

function sentimentBadgeVariant(sentiment: string): "success" | "warning" | "destructive" {
  const s = sentiment.toLowerCase();
  if (s === "positive" || s === "positiv") return "success";
  if (s === "negative" || s === "negativ") return "destructive";
  return "warning";
}

// ---------------------------------------------------------------------------
// EmotionCard — multi-label emotions per segment
// ---------------------------------------------------------------------------

export function EmotionCard({ emotion }: { emotion: EmotionSegmentResult[] }) {
  if (emotion.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Heart className="size-4" />
          Känslolabels
        </CardTitle>
        <CardDescription>
          Granulära känslolabels per segment (frustration, ilska, besvikelse, etc.).
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {emotion.slice(0, 12).map((e, i) => {
          const topScore = e.scores ? Math.max(...Object.values(e.scores)) : 0;
          return (
            <div key={i} className="flex items-center justify-between gap-2 rounded-md border p-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">#{i}</span>
                <Badge variant="secondary">{EMOTION_LABELS_SV[e.primary] ?? e.primary}</Badge>
                {e.speaker && (
                  <span className="text-xs text-muted-foreground">{e.speaker}</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {Object.entries(e.scores)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([label, score]) => (
                    <span key={label} className="text-xs text-muted-foreground">
                      {EMOTION_LABELS_SV[label] ?? label}: {(score * 100).toFixed(0)}%
                    </span>
                  ))}
                <span className={`text-xs font-medium ${scoreColor(topScore * 100)}`}>
                  {(topScore * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
        {emotion.length > 12 && (
          <span className="text-xs text-muted-foreground">
            +{emotion.length - 12} fler segment…
          </span>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// AspectCard — aspect-based sentiment analysis
// ---------------------------------------------------------------------------

export function AspectCard({
  aspects,
  derivedSentiment,
}: {
  aspects: AspectItem[];
  derivedSentiment?: DerivedCallSentiment | null;
}) {
  if (aspects.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="size-4" />
          Aspekt-evidens (claim charts)
        </CardTitle>
        <CardDescription>
          Primär produktenhet: aspekter med citat. Call-sentiment är härlett aggregat.
          {derivedSentiment && (
            <>
              {" "}
              Härlett: <strong>{derivedSentiment.label}</strong> (
              {derivedSentiment.score.toFixed(2)}, {derivedSentiment.aspect_count} aspekter)
            </>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {aspects.slice(0, 10).map((a, i) => {
          const quote =
            a.evidence_spans?.[0]?.text ||
            a.evidence ||
            null;
          return (
            <div key={i} className="flex flex-col gap-1 rounded-md border p-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium capitalize">
                  {a.aspect.replace(/_/g, " ")}
                </span>
                <div className="flex items-center gap-1">
                  {a.source && (
                    <Badge variant="outline">{a.source === "llm_refined" ? "LLM" : "ABSA"}</Badge>
                  )}
                  <Badge variant={sentimentBadgeVariant(a.sentiment)}>{a.sentiment}</Badge>
                </div>
              </div>
              {quote && (
                <span className="text-xs text-muted-foreground italic">&ldquo;{quote}&rdquo;</span>
              )}
              {a.speaker && (
                <span className="text-xs text-muted-foreground">Talare: {a.speaker}</span>
              )}
            </div>
          );
        })}
        {aspects.length > 10 && (
          <span className="text-xs text-muted-foreground">+{aspects.length - 10} fler aspekter…</span>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// TrajectoryCard — conversation trajectory & escalation
// ---------------------------------------------------------------------------

export function TrajectoryCard({ trajectory }: { trajectory: TrajectoryResult | null }) {
  if (!trajectory) return null;

  const slope = trajectory.customer_sentiment_slope;
  const slopeIcon = slope > 0.02 ? <TrendingUp className="size-4 text-success" /> :
    slope < -0.02 ? <TrendingDown className="size-4 text-destructive" /> :
    <Minus className="size-4 text-muted-foreground" />;
  const slopeLabel = slope > 0.02 ? "Förbättras" : slope < -0.02 ? "Försämras" : "Stabil";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Route className="size-4" />
          Samtalstrajectory
        </CardTitle>
        <CardDescription>Sentimentkurva, eskalationshändelser och frustrationstopp.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Sentimentlutning</span>
            <span className="flex items-center gap-1.5 text-sm font-medium">
              {slopeIcon} {slopeLabel}
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Eskalationer</span>
            <span className="text-sm font-medium">{trajectory.escalation_events}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Frustrationstopp</span>
            <span className="text-sm font-medium">
              {trajectory.peak_frustration_turn !== null
                ? `Segment #${trajectory.peak_frustration_turn}`
                : "Ingen"}
            </span>
          </div>
        </div>

        {trajectory.sentiment_trend.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Sentiment över tid</span>
            <div className="flex h-12 items-end gap-0.5">
              {trajectory.sentiment_trend.map((s, i) => {
                const normalized = Math.min(1, Math.max(0, (s + 1) / 2));
                return (
                  <div
                    key={i}
                    className={`flex-1 rounded-sm ${
                      s > 0 ? "bg-success/70" : s < 0 ? "bg-destructive/70" : "bg-muted"
                    }`}
                    style={{ height: `${normalized * 100}%` }}
                    title={`Segment ${i}: ${s.toFixed(2)}`}
                  />
                );
              })}
            </div>
          </div>
        )}

        {trajectory.escalation_event_details.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Eskalationsdetaljer</span>
            {trajectory.escalation_event_details.slice(0, 5).map((e, i) => {
              const emotion = e.emotion ? String(e.emotion) : null;
              const evidence = e.evidence ? String(e.evidence) : null;
              return (
                <div key={i} className="flex items-center gap-2 rounded-md border p-1.5 text-xs">
                  <AlertTriangle className="size-3 shrink-0 text-warning" />
                  <span className="text-muted-foreground">#{String(e.turn ?? i)}</span>
                  <span className="capitalize">{String(e.type ?? "").replace(/_/g, " ")}</span>
                  {emotion && <Badge variant="warning" className="text-xs">{emotion}</Badge>}
                  {evidence && (
                    <span className="truncate italic text-muted-foreground">&ldquo;{evidence}&rdquo;</span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// RootCauseCard — root cause analysis
// ---------------------------------------------------------------------------

export function RootCauseCard({ rootCause }: { rootCause: RootCauseResult | null }) {
  if (!rootCause) return null;
  const hasCauses = rootCause.root_causes && rootCause.root_causes.length > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="size-4" />
          Rotorsaksanalys
        </CardTitle>
        <CardDescription>
          Identifierade bakomliggande orsaker till kundens problem.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Risknivå</span>
          <Badge variant={riskBadgeVariant(rootCause.overall_risk)}>{rootCause.overall_risk}</Badge>
        </div>

        {rootCause.top_root_cause && (
          <div className="rounded-md bg-muted/50 p-2.5">
            <span className="text-xs text-muted-foreground">Primär orsak</span>
            <p className="text-sm font-medium capitalize">{rootCause.top_root_cause}</p>
          </div>
        )}

        {hasCauses && (
          <div className="flex flex-col gap-1.5">
            {rootCause.root_causes.map((c, i) => (
              <div key={i} className="flex flex-col gap-1 rounded-md border p-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium capitalize">{c.cause}</span>
                  <Badge variant="outline" className="text-xs">{c.count}× träff</Badge>
                </div>
                {c.recommendation && (
                  <span className="text-xs text-muted-foreground">{c.recommendation}</span>
                )}
              </div>
            ))}
          </div>
        )}

        {!hasCauses && rootCause.message && (
          <p className="text-sm text-muted-foreground">{rootCause.message}</p>
        )}

        {rootCause.evidence_examples.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Bevis</span>
            {rootCause.evidence_examples.slice(0, 3).map((e, i) => (
              <span key={i} className="text-xs italic text-muted-foreground">
                &ldquo;{String(e.evidence ?? "")}&rdquo;
              </span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// CoachingCard — actionable coaching recommendations
// ---------------------------------------------------------------------------

export function CoachingCard({ coaching }: { coaching: CoachingResult | null }) {
  if (!coaching || coaching.insight_count === 0) return null;

  const priorityVariant: Record<string, "destructive" | "warning" | "secondary"> = {
    high: "destructive",
    medium: "warning",
    low: "secondary",
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lightbulb className="size-4" />
          Coaching-rekommendationer
        </CardTitle>
        <CardDescription>
          Prioriterade åtgärder baserat på samtalets analyser.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {coaching.top_recommendation && (
          <div className="rounded-md border border-primary/30 bg-primary/5 p-3">
            <span className="text-xs font-medium text-primary">Topp-rekommendation</span>
            <p className="text-sm">{coaching.top_recommendation}</p>
          </div>
        )}

        {coaching.coaching_insights.length > 0 && (
          <div className="flex flex-col gap-1.5">
            {coaching.coaching_insights.map((insight, i) => (
              <div key={i} className="flex flex-col gap-1 rounded-md border p-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-mono text-muted-foreground">{insight.rule_id}</span>
                  <Badge variant={priorityVariant[insight.priority] ?? "secondary"} className="text-xs">
                    {insight.priority}
                  </Badge>
                </div>
                <span className="text-sm">{insight.recommendation}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// CustomerEffortCard — Customer Effort Score (CES)
// ---------------------------------------------------------------------------

export function CustomerEffortCard({ effort }: { effort: CustomerEffortResult | null }) {
  if (!effort) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Gauge className="size-4" />
          Kundinsats (CES)
        </CardTitle>
        <CardDescription>
          Customer Effort Score — högre värde indikerar mer friktion för kunden.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">CES-score</span>
            <span className={`text-2xl font-semibold ${scoreColor(100 - effort.overall_ces)}`}>
              {Math.round(effort.overall_ces)}/100
            </span>
          </div>
          <span className="text-xs text-muted-foreground">{effort.scale}</span>
        </div>

        {effort.per_segment.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Per segment (topp 5)</span>
            {effort.per_segment
              .slice()
              .sort((a, b) => b.effort_score - a.effort_score)
              .slice(0, 5)
              .map((s, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-16 text-xs text-muted-foreground">{s.speaker ?? "?"}</span>
                  <Progress value={s.effort_score} className="flex-1" />
                  <span className="w-10 text-right text-xs">{s.effort_score.toFixed(0)}</span>
                </div>
              ))}
          </div>
        )}

        {effort.coaching_tips.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Coaching-tips</span>
            {effort.coaching_tips.map((tip, i) => (
              <span key={i} className="text-xs text-muted-foreground">• {tip}</span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// ActiveListeningCard — listening behaviors
// ---------------------------------------------------------------------------

export function ActiveListeningCard({ listening }: { listening: ActiveListeningResult | null }) {
  if (!listening) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Ear className="size-4" />
          Aktivt lyssnande
        </CardTitle>
        <CardDescription>
          Backchannels, avbrott och tal-/lyssnarbalans.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Lyssnarpoäng</span>
            <span className={`text-2xl font-semibold ${scoreColor(listening.listening_score)}`}>
              {Math.round(listening.listening_score)}/100
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Backchannels</span>
            <span className="text-2xl font-semibold">{listening.backchannel_count}</span>
          </div>
        </div>

        {Object.keys(listening.speaker_balance).length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Talbalans</span>
            {Object.entries(listening.speaker_balance).map(([speaker, pct]) => (
              <div key={speaker} className="flex items-center gap-2">
                <span className="w-20 text-xs text-muted-foreground">{speaker}</span>
                <Progress value={pct} className="flex-1" />
                <span className="w-12 text-right text-xs">{pct.toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}

        {listening.events.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Händelser</span>
            {listening.events.slice(0, 5).map((e, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                {e.type === "backchannel" ? (
                  <CheckCircle2 className="size-3 text-success" />
                ) : (
                  <XCircle className="size-3 text-destructive" />
                )}
                <span className="capitalize">{String(e.type).replace(/_/g, " ")}</span>
                <span className="text-muted-foreground">{String(e.speaker ?? "")}</span>
              </div>
            ))}
          </div>
        )}

        {listening.tips.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Tips</span>
            {listening.tips.map((tip, i) => (
              <span key={i} className="text-xs text-muted-foreground">• {tip}</span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// EmpathyCard — per-segment empathy with coaching tips
// ---------------------------------------------------------------------------

export function EmpathyCard({ empathy }: { empathy: EmpathyResult | null }) {
  if (!empathy) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Heart className="size-4" />
          Empati
        </CardTitle>
        <CardDescription>
          Empatipoäng per segment med coaching-tips.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Övergripande empati</span>
            <span className={`text-2xl font-semibold ${scoreColor(empathy.overall_empathy)}`}>
              {Math.round(empathy.overall_empathy)}/100
            </span>
          </div>
          <span className="text-xs text-muted-foreground">{empathy.scale}</span>
        </div>

        {empathy.per_segment.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Per segment (topp 5)</span>
            {empathy.per_segment
              .slice()
              .sort((a, b) => b.empathy_score - a.empathy_score)
              .slice(0, 5)
              .map((s, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-16 text-xs text-muted-foreground">{s.speaker ?? "?"}</span>
                  <Progress value={s.empathy_score} className="flex-1" />
                  <span className="w-10 text-right text-xs">{s.empathy_score.toFixed(0)}</span>
                </div>
              ))}
          </div>
        )}

        {empathy.coaching_tips.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Coaching-tips</span>
            {empathy.coaching_tips.map((tip, i) => (
              <span key={i} className="text-xs text-muted-foreground">• {tip}</span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// ResolutionProbabilityCard — resolution probability
// ---------------------------------------------------------------------------

export function ResolutionProbabilityCard({
  resolution,
}: {
  resolution: ResolutionProbabilityResult | null;
}) {
  if (!resolution) return null;

  const prob = resolution.resolution_probability;
  const probVariant = prob > 70 ? "success" : prob < 45 ? "destructive" : "warning";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CheckCircle2 className="size-4" />
          Lösnings sannolikhet
        </CardTitle>
        <CardDescription>
          Uppskattad sannolikhet att ärendet löses i detta samtal.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Sannolikhet</span>
            <span className={`text-2xl font-semibold ${scoreColor(prob)}`}>
              {Math.round(prob)}%
            </span>
          </div>
          <Badge variant={probVariant}>Konfidens: {resolution.confidence}%</Badge>
        </div>

        <div className="rounded-md bg-muted/50 p-2.5">
          <span className="text-xs text-muted-foreground">Rekommenderad åtgärd</span>
          <p className="text-sm">{resolution.recommended_action}</p>
        </div>

        {resolution.factors && Object.keys(resolution.factors).length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Faktorer</span>
            {Object.entries(resolution.factors).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between text-xs">
                <span className="capitalize text-muted-foreground">{key.replace(/_/g, " ")}</span>
                <span className="font-medium">{String(value)}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// JourneyCard — multi-turn journey
// ---------------------------------------------------------------------------

const STAGE_COLORS: Record<string, "success" | "warning" | "destructive" | "secondary"> = {
  opening: "secondary",
  escalation: "destructive",
  resolution: "success",
};

export function JourneyCard({ journey }: { journey: MultiTurnJourneyResult | null }) {
  if (!journey) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Route className="size-4" />
          Kundens resa (multi-turn)
        </CardTitle>
        <CardDescription>
          Hur samtalet utvecklas från öppning till resolution.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          {journey.resolved ? (
            <Badge variant="success">
              <CheckCircle2 className="size-3" /> Löst
            </Badge>
          ) : (
            <Badge variant="warning">
              <XCircle className="size-3" /> Ej löst
            </Badge>
          )}
          {journey.unresolved_count > 0 && (
            <span className="text-xs text-muted-foreground">
              {journey.unresolved_count} olösta eskalationer
            </span>
          )}
        </div>

        {journey.journey_stages.length > 0 && (
          <div className="flex flex-col gap-1">
            {journey.journey_stages.slice(0, 10).map((s, i) => (
              <div key={i} className="flex items-center gap-2 rounded-md border p-1.5 text-xs">
                <span className="w-8 text-muted-foreground">#{i}</span>
                <Badge variant={STAGE_COLORS[s.stage] ?? "secondary"} className="capitalize">
                  {s.stage}
                </Badge>
                <span className="truncate italic text-muted-foreground">{s.text_snippet}</span>
              </div>
            ))}
          </div>
        )}

        {journey.recommendation && (
          <p className="text-xs text-muted-foreground">{journey.recommendation}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// UpsellCard — upsell opportunities
// ---------------------------------------------------------------------------

export function UpsellCard({ upsell }: { upsell: UpsellResult | null }) {
  if (!upsell || upsell.count === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ArrowUpCircle className="size-4" />
          Upsell-möjligheter
        </CardTitle>
        <CardDescription>
          Tillfällen där upsell/cross-sell är naturligt och lågfriktion.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {upsell.opportunities.slice(0, 5).map((o, i) => (
          <div key={i} className="flex flex-col gap-1 rounded-md border p-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">{o.speaker ?? "?"}</span>
              <Badge variant="success" className="text-xs">{o.confidence.toFixed(0)}% confidence</Badge>
            </div>
            {o.evidence && (
              <span className="text-xs italic text-muted-foreground">&ldquo;{o.evidence}&rdquo;</span>
            )}
            {o.suggested_action && (
              <span className="text-xs">{o.suggested_action}</span>
            )}
            {o.signals.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {o.signals.map((sig, j) => (
                  <Badge key={j} variant="outline" className="text-xs">{sig}</Badge>
                ))}
              </div>
            )}
          </div>
        ))}
        {upsell.recommendation && (
          <p className="text-xs text-muted-foreground">{upsell.recommendation}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// RoleMetricsCard — role classifier extended metrics
// ---------------------------------------------------------------------------

export function RoleMetricsCard({ role }: { role: RoleClassifierResult | null }) {
  if (!role) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="size-4" />
          Rollanalys
        </CardTitle>
        <CardDescription>
          Talbalans, frågetäthet, lexikal formalitet och interventioner.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3">
          {role.talk_ratios && (
            <>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">Agent talandel</span>
                <span className="text-sm font-medium">{(role.talk_ratios.agent ?? 0).toFixed(0)}%</span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">Kund talandel</span>
                <span className="text-sm font-medium">{(role.talk_ratios.customer ?? 0).toFixed(0)}%</span>
              </div>
            </>
          )}
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Frågetäthet (agent)</span>
            <span className="text-sm font-medium">{role.question_density?.agent?.toFixed(2) ?? "0"}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Interventioner</span>
            <span className="text-sm font-medium">{role.intervention_count}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Formalitet</span>
            <span className="text-sm font-medium">{role.lexical_formality.toFixed(2)}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Agentens turer</span>
            <span className="text-sm font-medium">{role.num_agent_turns}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Kundens turer</span>
            <span className="text-sm font-medium">{role.num_customer_turns}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// PredictiveCard — predictive risk details
// ---------------------------------------------------------------------------

export function PredictiveCard({ predictive }: { predictive: PredictiveResult | null }) {
  if (!predictive) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="size-4" />
          Prediktiv risk
        </CardTitle>
        <CardDescription>
          Churn-risk, eskaleringsrisk och tillfredsställelse.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Risknivå</span>
          <Badge variant={riskBadgeVariant(predictive.risk_level)}>{predictive.risk_level}</Badge>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Churn-risk</span>
              <span className="text-xs font-medium">{(predictive.churn_risk * 100).toFixed(0)}%</span>
            </div>
            <Progress value={predictive.churn_risk * 100} indicatorClassName="bg-destructive" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Eskaleringsrisk</span>
              <span className="text-xs font-medium">{(predictive.escalation_risk * 100).toFixed(0)}%</span>
            </div>
            <Progress value={predictive.escalation_risk * 100} indicatorClassName="bg-warning" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Tillfredsställelse</span>
              <span className="text-xs font-medium">{(predictive.satisfaction_score * 100).toFixed(0)}%</span>
            </div>
            <Progress value={predictive.satisfaction_score * 100} indicatorClassName="bg-success" />
          </div>
        </div>

        {predictive.risk_factors.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Riskfaktorer</span>
            <div className="flex flex-wrap gap-1">
              {predictive.risk_factors.map((f, i) => (
                <Badge key={i} variant="outline" className="text-xs">{f}</Badge>
              ))}
            </div>
          </div>
        )}

        {predictive.recommended_action && (
          <div className="rounded-md bg-muted/50 p-2.5">
            <span className="text-xs text-muted-foreground">Rekommenderad åtgärd</span>
            <p className="text-sm">{predictive.recommended_action}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// ComplianceRiskCard — detailed compliance risk
// ---------------------------------------------------------------------------

export function ComplianceRiskCard({ compliance }: { compliance: ComplianceRiskResult | null }) {
  if (!compliance) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="size-4" />
          Compliance-risk (detaljerad)
        </CardTitle>
        <CardDescription>
          Compliance, legal och policy-risk i agentens språk.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Övergripande risk</span>
          <Badge variant={riskBadgeVariant(compliance.overall_risk_level)}>
            {compliance.overall_risk_level}
          </Badge>
        </div>

        {compliance.flagged_segments.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-muted-foreground">Flaggade segment</span>
            {compliance.flagged_segments.slice(0, 5).map((s, i) => {
              const severity = s.severity ? String(s.severity) : null;
              const evidence = s.evidence ? String(s.evidence) : null;
              const risks = Array.isArray(s.risks) ? s.risks : [];
              return (
                <div key={i} className="flex flex-col gap-1 rounded-md border p-2 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-muted-foreground">
                      {String(s.speaker ?? "")} @ {String(s.start ?? "")}s
                    </span>
                    {severity && (
                      <Badge variant={riskBadgeVariant(severity)} className="text-xs">
                        {severity}
                      </Badge>
                    )}
                  </div>
                  {evidence && (
                    <span className="italic text-muted-foreground">&ldquo;{evidence}&rdquo;</span>
                  )}
                  {risks.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {risks.map((r: unknown, j: number) => (
                        <Badge key={j} variant="outline" className="text-xs">{String(r)}</Badge>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {compliance.recommendation && (
          <p className="text-xs text-muted-foreground">{compliance.recommendation}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// SummaryCard — call summary (uses top-level summary field)
// ---------------------------------------------------------------------------

export function SummaryCard({ summary }: { summary: Record<string, unknown> | null }) {
  if (!summary) return null;
  const text = typeof summary.summary === "string" ? summary.summary : null;
  const actionItems = Array.isArray(summary.action_items) ? summary.action_items : [];

  if (!text && actionItems.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="size-4" />
          Sammanfattning
        </CardTitle>
        <CardDescription>Sammanfattning och åtgärdspunkter från samtalet.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {text && <p className="text-sm">{text}</p>}
        {actionItems.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Åtgärdspunkter</span>
            {actionItems.map((item, i) => (
              <span key={i} className="text-sm text-muted-foreground">
                • {typeof item === "string" ? item : String((item as Record<string, unknown>)?.text ?? item)}
              </span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// TrustSurfaceCard — CCP, degradation, routing, provenance (gap-plan Wave 1)
// ---------------------------------------------------------------------------

export interface TrustSurfaceData {
  degradation: DegradationInfo | null;
  deepPathCCP: DeepPathCCP | null;
  analyzerRouting: AnalyzerRouting | null;
  overrideProvenance: OverrideProvenanceEntry[];
}

export function TrustSurfaceCard({ trust }: { trust: TrustSurfaceData | null }) {
  if (!trust) return null;
  const { degradation, deepPathCCP, analyzerRouting, overrideProvenance } = trust;
  const showDegradation =
    degradation &&
    (degradation.mode === "honest" || (!degradation.llm_used && degradation.deep_path_active === false));
  const hasCCP = deepPathCCP && deepPathCCP.checks.length > 0;
  const hasRouting = analyzerRouting && (analyzerRouting.runtime_selected?.length ?? 0) > 0;
  const hasProvenance = overrideProvenance.length > 0;
  if (!showDegradation && !hasCCP && !hasRouting && !hasProvenance) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="size-4" />
          Trust & routing
        </CardTitle>
        <CardDescription>
          Varför deep path kördes eller hoppades över, samt vilka analyzers som valdes.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {showDegradation ? (
          <div className="rounded-md border border-warning/40 bg-warning/10 p-3 text-sm">
            <div className="flex items-center gap-2 font-medium text-warning-text">
              <AlertTriangle className="size-4 shrink-0" />
              Honest degradation
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Deep path: {degradation.deep_path_active ? "begärd" : "av"} · LLM använd:{" "}
              {degradation.llm_used ? "ja" : "nej"} · Vissa fält kan vara{" "}
              <code>unavailable</code> (kräver deep path).
            </p>
          </div>
        ) : null}

        {hasCCP ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">Deep-path CCP</span>
              <Badge variant={deepPathCCP.passed ? "success" : "destructive"}>
                {deepPathCCP.passed ? "Godkänd" : "Underkänd"}
              </Badge>
            </div>
            <div className="flex flex-col gap-1.5">
              {deepPathCCP.checks.map((check) => (
                <div
                  key={check.name}
                  className="flex items-start justify-between gap-2 rounded-md border p-2 text-xs"
                >
                  <div className="flex flex-col gap-0.5">
                    <span className="font-medium">{check.name}</span>
                    <span className="text-muted-foreground">{check.detail}</span>
                    {check.corrective_action ? (
                      <span className="text-warning-text">{check.corrective_action}</span>
                    ) : null}
                  </div>
                  {check.passed ? (
                    <CheckCircle2 className="size-4 shrink-0 text-success" />
                  ) : (
                    <XCircle className="size-4 shrink-0 text-destructive" />
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {hasRouting ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <GitBranch className="size-3.5" />
              Analyzer routing
              {analyzerRouting.applied ? (
                <Badge variant="secondary" className="text-xs">
                  applied
                </Badge>
              ) : null}
            </div>
            {analyzerRouting.profile_prior && analyzerRouting.profile_prior.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                <span className="text-xs text-muted-foreground">Prior:</span>
                {analyzerRouting.profile_prior.map((a) => (
                  <Badge key={`prior-${a}`} variant="outline" className="text-xs">
                    {a}
                  </Badge>
                ))}
              </div>
            ) : null}
            {analyzerRouting.runtime_selected && analyzerRouting.runtime_selected.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                <span className="text-xs text-muted-foreground">Runtime:</span>
                {analyzerRouting.runtime_selected.map((a) => (
                  <Badge key={`rt-${a}`} variant="secondary" className="text-xs">
                    {a}
                  </Badge>
                ))}
              </div>
            ) : null}
            {analyzerRouting.extras_run && analyzerRouting.extras_run.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                <span className="text-xs text-muted-foreground">Extras:</span>
                {analyzerRouting.extras_run.map((a) => (
                  <Badge key={`ex-${a}`} variant="outline" className="text-xs">
                    {a}
                  </Badge>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {hasProvenance ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <History className="size-3.5" />
              LLM override provenance
            </div>
            {overrideProvenance.slice(0, 6).map((entry, i) => (
              <div key={`${entry.field}-${i}`} className="rounded-md border p-2 text-xs">
                <span className="font-medium">{entry.field}</span>
                {entry.source ? (
                  <span className="text-muted-foreground"> · {entry.source}</span>
                ) : null}
                {entry.evidence_spans && entry.evidence_spans.length > 0 ? (
                  <p className="mt-1 text-muted-foreground">
                    &ldquo;{entry.evidence_spans[0]?.text}&rdquo;
                  </p>
                ) : null}
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
