"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { ArrowLeft, CheckCircle2, XCircle, Quote, FileSearch } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { EmptyState } from "@/components/empty-state";
import { EmotionTimelineChart } from "@/components/emotion-timeline-chart";
import { RiskBadge, SentimentBadge } from "@/components/status-badges";
import { CallAlertsSection } from "@/components/call-alerts-section";
import { LlmJudgePanel } from "@/components/llm-judge-panel";
import { TranscriptView } from "@/components/transcript-view";
import {
  EmotionCard,
  AspectCard,
  TrajectoryCard,
  RootCauseCard,
  CoachingCard,
  CustomerEffortCard,
  ActiveListeningCard,
  EmpathyCard,
  ResolutionProbabilityCard,
  JourneyCard,
  UpsellCard,
  RoleMetricsCard,
  PredictiveCard,
  ComplianceRiskCard,
  SummaryCard,
} from "@/components/analyzer-cards";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { buildCallDetail } from "@/lib/real-data";

export default function CallDetailPage() {
  const params = useParams<{ id: string }>();
  const { reports, isLoading, isError } = useDemoReports();

  const realCall = reports.find((r) => r.transcript.id === params.id);

  if (!isLoading && !realCall && !isError) {
    notFound();
  }

  const detail = realCall ? buildCallDetail(realCall) : null;
  const callRow = realCall
    ? {
        title: realCall.transcript.title,
        agent: realCall.transcript.meta.agent,
        durationS: realCall.transcript.meta.duration_s,
        category: realCall.transcript.meta.category,
      }
    : null;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" asChild aria-label="Tillbaka">
          <Link href="/">
            <ArrowLeft className="size-4" />
          </Link>
        </Button>
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold tracking-tight">
            {callRow?.title ?? "Laddar…"}
          </h1>
          <p className="text-sm text-muted-foreground">
            {params.id} · {callRow?.agent ?? ""} · {callRow ? Math.round(callRow.durationS / 60) : 0} min
          </p>
        </div>
      </div>

      {isLoading ? (
        <Card>
          <CardContent className="pt-5">
            <EmptyState
              icon={FileSearch}
              title="Laddar samtalsdata…"
              hint="Hämtar pipeline-resultat från backend. Detta kan ta några sekunder."
            />
          </CardContent>
        </Card>
      ) : isError || !detail ? (
        <Card>
          <CardContent className="pt-5">
            <EmptyState
              icon={FileSearch}
              title="Kunde inte hämta samtalsdata"
              hint="Kontrollera att backend-API:et körs och är nåbart från webui."
            />
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <SentimentBadge value={detail.qa ? (detail.qa.score >= 60 ? "positive" : detail.qa.score >= 40 ? "neutral" : "negative") : "neutral"} />
            <RiskBadge value={detail.qa?.riskLevel ?? "medium"} />
            <Badge variant="outline" className="capitalize">
              {callRow?.category}
            </Badge>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="flex flex-col gap-4 lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Transkript</CardTitle>
                  {detail.transcript.length > 50 && (
                    <CardDescription>
                      Virtualiserad vy · {detail.transcript.length} segment
                    </CardDescription>
                  )}
                </CardHeader>
                <CardContent>
                  <TranscriptView turns={detail.transcript} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Känslotidslinje</CardTitle>
                  <CardDescription>Sentiment över samtalets längd.</CardDescription>
                </CardHeader>
                <CardContent>
                  {detail.emotionTimeline.length > 0 ? (
                    <EmotionTimelineChart data={detail.emotionTimeline} />
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Ingen sentimentdata att visa.
                    </p>
                  )}
                </CardContent>
              </Card>

              <LlmJudgePanel result={detail.llmJudge} />

              {/* Fas 5: Analyzer cards — agent coaching + core analyzers */}
              <div className="flex flex-col gap-4">
                <h2 className="text-sm font-semibold tracking-tight text-muted-foreground">
                  Analysdetaljer
                </h2>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <EmotionCard emotion={detail.emotion} />
                  <AspectCard
                    aspects={detail.aspects}
                    derivedSentiment={detail.derivedCallSentiment}
                  />
                  <TrajectoryCard trajectory={detail.trajectory} />
                  <RootCauseCard rootCause={detail.rootCause} />
                  <CoachingCard coaching={detail.coaching} />
                  <CustomerEffortCard effort={detail.customerEffort} />
                  <ActiveListeningCard listening={detail.activeListening} />
                  <EmpathyCard empathy={detail.empathy} />
                  <ResolutionProbabilityCard resolution={detail.resolutionProbability} />
                  <JourneyCard journey={detail.journey} />
                  <UpsellCard upsell={detail.upsell} />
                  <RoleMetricsCard role={detail.roleMetrics} />
                  <PredictiveCard predictive={detail.predictive} />
                  <ComplianceRiskCard compliance={detail.complianceRisk} />
                  <SummaryCard summary={(realCall?.report.summary as Record<string, unknown> | null) ?? null} />
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-4">
              {detail.qa && (
                <Card>
                  <CardHeader>
                    <CardTitle>QA & Compliance</CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="flex flex-col gap-1">
                        <span className="text-xs text-muted-foreground">QA-poäng</span>
                        <span className="text-2xl font-semibold">{Math.round(detail.qa.score)}/100</span>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-xs text-muted-foreground">Status</span>
                        <Badge variant={detail.qa.passed ? "success" : "destructive"} className="w-fit">
                          {detail.qa.passed ? "Godkänd" : "Underkänd"}
                        </Badge>
                      </div>
                    </div>

                    {detail.qa.complianceFlags.length > 0 ? (
                      <div className="flex flex-col gap-1.5">
                        <span className="text-xs font-medium text-muted-foreground">
                          Compliance-flaggor
                        </span>
                        <div className="flex flex-wrap gap-1.5">
                          {detail.qa.complianceFlags.map((flag) => (
                            <Badge key={flag} variant="warning">
                              {flag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {detail.qa.criteria.length > 0 && (
                      <div className="flex flex-col gap-3">
                        <span className="text-xs font-medium text-muted-foreground">Kriterier</span>
                        {detail.qa.criteria.map((c, i) => (
                          <div key={i} className="flex flex-col gap-1">
                            <div className="flex items-center justify-between gap-2">
                              <span className="flex items-center gap-1.5 text-sm">
                                {c.passed ? (
                                  <CheckCircle2 className="size-4 shrink-0 text-success" />
                                ) : (
                                  <XCircle className="size-4 shrink-0 text-destructive" />
                                )}
                                {c.criterion}
                              </span>
                              <span className="shrink-0 text-xs text-muted-foreground">
                                {Math.round(c.score)}
                              </span>
                            </div>
                            <Progress
                              value={c.score}
                              indicatorClassName={c.passed ? "bg-success" : "bg-destructive"}
                            />
                            {c.evidence && (
                              <span className="text-xs text-muted-foreground">{c.evidence}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              <CallAlertsSection alerts={detail.alerts} />

              {detail.evidenceQuotes.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Beviscitat</CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-2">
                    {detail.evidenceQuotes.map((quote, i) => (
                      <div key={i} className="flex items-start gap-2 rounded-md bg-muted/50 p-2.5 text-sm">
                        <Quote className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                        <span>{quote}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
