"use client";

import * as React from "react";
import Link from "next/link";
import { Microscope, FileSearch, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { EmptyState } from "@/components/empty-state";
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
import { RiskBadge, SentimentBadge } from "@/components/status-badges";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { buildCallDetail, getOverallSentiment } from "@/lib/real-data";

export default function AnalysisPage() {
  const { reports, isLoading, isError } = useDemoReports();
  const [explicitId, setExplicitId] = React.useState<string | null>(null);

  // Derive the effective selected call: explicit selection > first available
  const effectiveId = explicitId ?? (reports.length > 0 ? reports[0].transcript.id : null);
  const selectedCall = reports.find((r) => r.transcript.id === effectiveId) ?? null;
  const detail = selectedCall ? buildCallDetail(selectedCall) : null;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">
          <Microscope className="size-5" />
          Analysdetaljer
        </h1>
        <p className="text-sm text-muted-foreground">
          Alla analysmetoder per samtal — känslolabels, aspekter, trajectory, rotdiagnos,
          coaching, kundinsats, aktivt lyssnande och mer.
        </p>
      </div>

      {/* Call selector */}
      <Card>
        <CardHeader>
          <CardTitle>Välj samtal</CardTitle>
          <CardDescription>
            Välj ett samtal för att se alla analysresultat i detalj.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-sm text-muted-foreground">Laddar samtal…</p>
          ) : isError ? (
            <EmptyState
              icon={FileSearch}
              title="Kunde inte hämta samtalsdata"
              hint="Kontrollera att backend-API:et körs."
            />
          ) : reports.length === 0 ? (
            <EmptyState
              icon={FileSearch}
              title="Inga samtal tillgängliga"
              hint="Kör pipelinen mot demo-transkript för att se resultat."
            />
          ) : (
            <Select
              value={effectiveId ?? undefined}
              onValueChange={setExplicitId}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Välj samtal…" />
              </SelectTrigger>
              <SelectContent>
                {reports.map((r) => {
                  const overall = getOverallSentiment(r.report);
                  return (
                    <SelectItem key={r.transcript.id} value={r.transcript.id}>
                      {r.transcript.title} · {r.transcript.meta.agent} · {overall.label}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          )}
        </CardContent>
      </Card>

      {/* Analyzer results */}
      {detail && selectedCall ? (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <SentimentBadge
              value={
                detail.qa
                  ? detail.qa.score >= 60
                    ? "positive"
                    : detail.qa.score >= 40
                      ? "neutral"
                      : "negative"
                  : "neutral"
              }
            />
            <RiskBadge value={detail.qa?.riskLevel ?? "medium"} />
            <Badge variant="outline" className="capitalize">
              {selectedCall.transcript.meta.category}
            </Badge>
            <Button variant="ghost" size="sm" asChild className="ml-auto gap-1">
              <Link href={`/calls/${selectedCall.transcript.id}`}>
                Öppna Call Detail <ChevronRight className="size-3.5" />
              </Link>
            </Button>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
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
            <SummaryCard summary={(selectedCall.report.summary as Record<string, unknown> | null) ?? null} />
          </div>
        </>
      ) : !isLoading && !isError && reports.length > 0 ? (
        <Card>
          <CardContent className="pt-5">
            <EmptyState
              icon={Microscope}
              title="Välj ett samtal ovan"
              hint="Alla analysresultat visas här när du väljer ett samtal."
            />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
