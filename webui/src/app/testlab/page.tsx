"use client";

import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { FlaskConical, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { EmptyState } from "@/components/empty-state";
import { ModelRoutingCard } from "@/components/model-routing-card";
import { ModelComparePanel } from "@/components/model-compare-panel";
import { apiClient, ApiError, type PipelineCompareResponse, type PipelineReport } from "@/lib/api/client";
import { extractTrustSurface } from "@/lib/real-data";
import { TrustSurfaceCard } from "@/components/analyzer-cards";
import { notifyApiError, notifySuccess } from "@/lib/notify";
import { useHealth } from "@/hooks/use-health";
import { type RoutingTier, resolveEffectiveTier, tierToModel } from "@/lib/routing-tier";

const EXAMPLE_SEGMENTS = JSON.stringify(
  [{ text: "Hej, hur kan jag hjälpa dig?", speaker: "Agent" }],
  null,
  2,
);

export default function TestLabPage() {
  const { data: connected } = useHealth();
  const [segmentsInput, setSegmentsInput] = React.useState("");
  const [useLlm, setUseLlm] = React.useState(false);
  const [provider, setProvider] = React.useState<"openrouter" | "groq">("openrouter");
  const [routingTier, setRoutingTier] = React.useState<RoutingTier>("balanced");
  const [partialMode, setPartialMode] = React.useState(false);
  const [reconcile, setReconcile] = React.useState(false);
  const [partialPrevious, setPartialPrevious] = React.useState<Record<string, unknown> | null>(null);

  const mutation = useMutation<PipelineReport, ApiError, void>({
    mutationFn: async () => {
      const raw = segmentsInput.trim();
      if (!raw) throw new ApiError("Ange segments som JSON");
      let segments: unknown;
      try {
        segments = JSON.parse(raw);
      } catch {
        throw new ApiError("Ogiltig JSON i segments-fältet");
      }
      if (!Array.isArray(segments) || segments.length === 0) {
        throw new ApiError("segments måste vara en icke-tom lista");
      }
      const segArr = segments as unknown[];
      const effectiveTier = resolveEffectiveTier(routingTier, segArr.length, useLlm);
      if (partialMode) {
        return apiClient.analyzePipelinePartial(segments, {
          previous_results: partialPrevious,
          reconcile,
          use_mistral_llm: useLlm,
          deep_analysis: useLlm,
          provider,
          llm_model: useLlm ? tierToModel(effectiveTier) : undefined,
        });
      }
      return apiClient.analyzePipeline(segments, {
        use_mistral_llm: useLlm,
        deep_analysis: useLlm,
        provider,
        llm_model: useLlm ? tierToModel(effectiveTier) : undefined,
      });
    },
    onSuccess: (data) => {
      notifySuccess(partialMode ? "Partial pipeline klar" : "Pipeline-analys klar");
      if (partialMode && !reconcile) {
        setPartialPrevious((data.results as Record<string, unknown>) ?? null);
      }
      if (reconcile) {
        setPartialPrevious(null);
      }
    },
    onError: (err) => notifyApiError(err, "Pipeline-fel: "),
  });

  const compareMutation = useMutation<PipelineCompareResponse, ApiError, void>({
    mutationFn: async () => {
      const raw = segmentsInput.trim();
      if (!raw) throw new ApiError("Ange segments som JSON");
      let segments: unknown;
      try {
        segments = JSON.parse(raw);
      } catch {
        throw new ApiError("Ogiltig JSON i segments-fältet");
      }
      if (!Array.isArray(segments) || segments.length === 0) {
        throw new ApiError("segments måste vara en icke-tom lista");
      }
      const models = (["fast", "balanced", "deep"] as RoutingTier[]).map(tierToModel);
      return apiClient.comparePipeline(segments, models, {
        provider,
        use_mistral_llm: true,
      });
    },
    onSuccess: () => notifySuccess("Modelljämförelse klar"),
    onError: (err) => notifyApiError(err, "Jämförelse-fel: "),
  });

  const report = mutation.data;
  const sentimentLabels = report?.sentiment_results?.slice(0, 5).map((s) => s.label ?? "?") ?? [];
  const qaScore = report?.results?.qa?.overall_qa_score;
  const actionableProblem = report?.llm?.actionable_summary?.problem;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Testlabb</h1>
          <p className="text-sm text-muted-foreground">
            Kör pipeline på JSON-segment direkt mot <code>/analyze_pipeline</code>.
          </p>
        </div>
        <Badge variant={connected ? "success" : "warning"}>
          {connected ? "API ansluten" : "API ej verifierad"}
        </Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Pipeline på JSON-segment</CardTitle>
          <CardDescription>
            Motsvarar &ldquo;Text &amp; pipeline&rdquo; i den gamla Testlabb-fliken.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="segments" className="text-xs font-medium text-muted-foreground">
              Klistra in segments (JSON)
            </label>
            <Textarea
              id="segments"
              rows={6}
              placeholder={EXAMPLE_SEGMENTS}
              value={segmentsInput}
              onChange={(e) => setSegmentsInput(e.target.value)}
              className="font-mono text-xs"
            />
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <Checkbox checked={useLlm} onCheckedChange={(v) => setUseLlm(v === true)} />
              Använd LLM deep analysis
            </label>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox checked={partialMode} onCheckedChange={(v) => setPartialMode(v === true)} />
              Partial path (<code>/analyze_pipeline/partial</code>)
            </label>
            {partialMode ? (
              <label className="flex items-center gap-2 text-sm">
                <Checkbox checked={reconcile} onCheckedChange={(v) => setReconcile(v === true)} />
                Reconcile (holistic LLM)
              </label>
            ) : null}
            {useLlm ? (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">LLM-provider</span>
                <Select value={provider} onValueChange={(v) => setProvider(v as typeof provider)}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openrouter">openrouter</SelectItem>
                    <SelectItem value="groq">groq</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            ) : null}
          </div>

          {useLlm && provider === "groq" ? (
            <div className="flex items-start gap-2 rounded-md border border-warning/40 bg-warning/10 p-3 text-xs text-warning-text">
              <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
              <span>
                Groq: US/Saudi-datacenter (ingen EU-hosting). Aktivera PII-redigering innan
                användning i produktion.
              </span>
            </div>
          ) : null}

          {useLlm && provider === "openrouter" ? (
            <ModelRoutingCard
              tier={routingTier}
              onTierChange={setRoutingTier}
              effectiveTier={resolveEffectiveTier(
                routingTier,
                (() => {
                  try {
                    const parsed = JSON.parse(segmentsInput);
                    return Array.isArray(parsed) ? parsed.length : 0;
                  } catch {
                    return 0;
                  }
                })(),
                useLlm,
              )}
            />
          ) : null}

          <div className="flex flex-wrap gap-2">
            <Button onClick={() => mutation.mutate()} disabled={mutation.isPending} className="gap-1.5">
              <FlaskConical className="size-4" />
              {mutation.isPending
                ? "Analyserar…"
                : partialMode
                  ? reconcile
                    ? "Reconcile (partial)"
                    : "Kör partial"
                  : "Analysera (pipeline)"}
            </Button>
            {partialMode && partialPrevious ? (
              <Button
                variant="outline"
                onClick={() => setPartialPrevious(null)}
                disabled={mutation.isPending}
              >
                Nollställ partial state
              </Button>
            ) : null}
            {useLlm && provider === "openrouter" ? (
              <Button
                variant="outline"
                onClick={() => compareMutation.mutate()}
                disabled={compareMutation.isPending || mutation.isPending}
                className="gap-1.5"
              >
                {compareMutation.isPending ? "Jämför…" : "Jämför FAST/BALANCED/DEEP"}
              </Button>
            ) : null}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Resultat</CardTitle>
        </CardHeader>
        <CardContent>
          {mutation.isIdle ? (
            <EmptyState
              icon={FlaskConical}
              title="Inga resultat ännu"
              hint="Klistra in segments och klicka på Analysera."
            />
          ) : mutation.isPending ? (
            <p className="text-sm text-muted-foreground">Kör pipeline via backend…</p>
          ) : mutation.isError ? (
            <div className="flex flex-col gap-2">
              <Badge variant="destructive" className="w-fit">
                API-fel
              </Badge>
              <p className="text-sm">{mutation.error.message}</p>
              {mutation.error.detail ? (
                <pre className="max-h-64 overflow-auto rounded-md bg-muted/50 p-3 text-xs">
                  {typeof mutation.error.detail === "string"
                    ? mutation.error.detail
                    : JSON.stringify(mutation.error.detail, null, 2)}
                </pre>
              ) : null}
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              <Badge variant="success" className="w-fit">
                Analys klar
              </Badge>
              {sentimentLabels.length > 0 ? (
                <p className="text-sm">
                  Sentiment (första segment): {sentimentLabels.join(", ")}
                </p>
              ) : null}
              {qaScore !== undefined && qaScore !== null ? (
                <p className="text-sm">QA-poäng: {qaScore}/100</p>
              ) : null}
              {actionableProblem ? (
                <div className="rounded-md bg-muted/50 p-3 text-sm">
                  <span className="font-medium">Problem: </span>
                  {actionableProblem}
                </div>
              ) : null}

              {/* Fas 5: Analyzer output summary */}
              {report?.analyzer_results ? (
                <div className="flex flex-col gap-2 rounded-md border p-3">
                  <span className="text-xs font-medium text-muted-foreground">
                    Analyzer-resultat (typed view)
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(report.analyzer_results)
                      .filter(([, v]) => v !== null && v !== undefined)
                      .map(([key]) => (
                        <Badge key={key} variant="secondary" className="text-xs">
                          {key}
                        </Badge>
                      ))}
                  </div>
                  {report.analyzer_results.emotion && Array.isArray(report.analyzer_results.emotion) && (
                    <p className="text-xs text-muted-foreground">
                      Känslolabels: {report.analyzer_results.emotion.map((e) => e.primary).join(", ")}
                    </p>
                  )}
                  {report.analyzer_results.aspect && Array.isArray(report.analyzer_results.aspect) && (
                    <p className="text-xs text-muted-foreground">
                      Aspekter: {report.analyzer_results.aspect.map((a) => a.aspect).join(", ")}
                    </p>
                  )}
                  {report.analyzer_results.trajectory && (
                    <p className="text-xs text-muted-foreground">
                      Trajectory: lutning {report.analyzer_results.trajectory.customer_sentiment_slope}, {report.analyzer_results.trajectory.escalation_events} eskalationer
                    </p>
                  )}
                  {report.analyzer_results.root_cause?.top_root_cause && (
                    <p className="text-xs text-muted-foreground">
                      Rotorsak: {report.analyzer_results.root_cause.top_root_cause}
                    </p>
                  )}
                  {report.analyzer_results.actionable_coaching?.top_recommendation && (
                    <p className="text-xs text-muted-foreground">
                      Coaching: {report.analyzer_results.actionable_coaching.top_recommendation}
                    </p>
                  )}
                  {report.analyzer_results.customer_effort && (
                    <p className="text-xs text-muted-foreground">
                      CES: {Math.round(report.analyzer_results.customer_effort.overall_ces)}/100
                    </p>
                  )}
                  {report.analyzer_results.empathy && (
                    <p className="text-xs text-muted-foreground">
                      Empati: {Math.round(report.analyzer_results.empathy.overall_empathy)}/100
                    </p>
                  )}
                  {report.analyzer_results.resolution_probability && (
                    <p className="text-xs text-muted-foreground">
                      Lösningsgrad: {Math.round(report.analyzer_results.resolution_probability.resolution_probability)}%
                    </p>
                  )}
                  {report.analyzer_results.upsell_opportunity && report.analyzer_results.upsell_opportunity.count > 0 && (
                    <p className="text-xs text-muted-foreground">
                      Upsell-möjligheter: {report.analyzer_results.upsell_opportunity.count}
                    </p>
                  )}
                </div>
              ) : null}

              {report ? <TrustSurfaceCard trust={extractTrustSurface(report)} /> : null}

              {report?.results?.partial && typeof report.results.partial === "object" ? (
                <div className="rounded-md border p-3 text-xs">
                  <span className="font-medium">Partial metadata</span>
                  <pre className="mt-2 overflow-auto text-muted-foreground">
                    {JSON.stringify(report.results.partial, null, 2)}
                  </pre>
                </div>
              ) : null}

              <details className="text-xs">
                <summary className="cursor-pointer text-muted-foreground">
                  Fullständigt svar (JSON)
                </summary>
                <pre className="mt-2 max-h-96 overflow-auto rounded-md bg-muted/50 p-3">
                  {JSON.stringify(report, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </CardContent>
      </Card>

      {useLlm && provider === "openrouter" ? (
        <ModelComparePanel
          data={compareMutation.data}
          isLoading={compareMutation.isPending}
        />
      ) : null}
    </div>
  );
}
