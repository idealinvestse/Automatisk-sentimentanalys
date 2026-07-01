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
import { MOCK_CALLS, MOCK_CALL_DETAILS } from "@/lib/mock-data";

export default function CallDetailPage() {
  const params = useParams<{ id: string }>();
  const call = MOCK_CALLS.find((c) => c.callId === params.id);
  const detail = MOCK_CALL_DETAILS[params.id];

  if (!call) {
    notFound();
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" asChild aria-label="Tillbaka">
          <Link href="/">
            <ArrowLeft className="size-4" />
          </Link>
        </Button>
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold tracking-tight">{call.title}</h1>
          <p className="text-sm text-muted-foreground">
            {call.callId} · {call.agent} · {Math.round(call.durationS / 60)} min
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <SentimentBadge value={call.sentiment} />
        <RiskBadge value={call.riskLevel} />
        <Badge variant="outline" className="capitalize">
          {call.category}
        </Badge>
      </div>

      {!detail ? (
        <Card>
          <CardContent className="pt-5">
            <EmptyState
              icon={FileSearch}
              title="Ingen djupdykningsdata för detta samtal"
              hint="Transkript, QA-scorecard och bevis genereras av pipelinen (/analyze_pipeline eller /qa/score) och saknas i demomockdatan för detta samtal."
            />
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="flex flex-col gap-4 lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Transkript</CardTitle>
              </CardHeader>
              <CardContent className="flex max-h-96 flex-col gap-3 overflow-y-auto">
                {detail.transcript.map((turn, i) => (
                  <div key={i} className="flex gap-3">
                    <span className="w-10 shrink-0 text-xs text-muted-foreground">{turn.start}s</span>
                    <div className="flex min-w-0 flex-col">
                      <span
                        className={
                          turn.speaker === "Agent"
                            ? "text-xs font-medium text-primary"
                            : "text-xs font-medium text-muted-foreground"
                        }
                      >
                        {turn.speaker}
                      </span>
                      <span className="text-sm">{turn.text}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Känslotidslinje</CardTitle>
                <CardDescription>Sentiment över samtalets längd.</CardDescription>
              </CardHeader>
              <CardContent>
                <EmotionTimelineChart data={detail.emotionTimeline} />
              </CardContent>
            </Card>
          </div>

          <div className="flex flex-col gap-4">
            <Card>
              <CardHeader>
                <CardTitle>QA & Compliance</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex flex-col gap-1">
                    <span className="text-xs text-muted-foreground">QA-poäng</span>
                    <span className="text-2xl font-semibold">{detail.qa.score}/100</span>
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

                <div className="flex flex-col gap-3">
                  <span className="text-xs font-medium text-muted-foreground">Kriterier</span>
                  {detail.qa.criteria.map((c) => (
                    <div key={c.criterion} className="flex flex-col gap-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="flex items-center gap-1.5 text-sm">
                          {c.passed ? (
                            <CheckCircle2 className="size-4 shrink-0 text-success" />
                          ) : (
                            <XCircle className="size-4 shrink-0 text-destructive" />
                          )}
                          {c.criterion}
                        </span>
                        <span className="shrink-0 text-xs text-muted-foreground">{c.score}</span>
                      </div>
                      <Progress
                        value={c.score}
                        indicatorClassName={c.passed ? "bg-success" : "bg-destructive"}
                      />
                      <span className="text-xs text-muted-foreground">{c.evidence}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

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
          </div>
        </div>
      )}
    </div>
  );
}
