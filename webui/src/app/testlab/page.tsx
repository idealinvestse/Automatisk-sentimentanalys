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
import { apiClient, ApiError, type PipelineReport } from "@/lib/api/client";
import { notifyApiError, notifySuccess } from "@/lib/notify";
import { useHealth } from "@/hooks/use-health";

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
      return apiClient.analyzePipeline(segments, {
        use_mistral_llm: useLlm,
        deep_analysis: useLlm,
        provider,
      });
    },
    onSuccess: () => notifySuccess("Pipeline-analys klar"),
    onError: (err) => notifyApiError(err, "Pipeline-fel: "),
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

          <div>
            <Button onClick={() => mutation.mutate()} disabled={mutation.isPending} className="gap-1.5">
              <FlaskConical className="size-4" />
              {mutation.isPending ? "Analyserar…" : "Analysera (pipeline)"}
            </Button>
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
    </div>
  );
}
