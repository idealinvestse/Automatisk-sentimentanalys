"use client";

import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ClipboardCheck, WifiOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import { apiClient, ApiError, type QAScoreResponse } from "@/lib/api/client";
import { notifyApiError } from "@/lib/notify";
import type { RealCall } from "@/lib/real-data";

export function QaScorePanel({ reports }: { reports: RealCall[] }) {
  const [selectedId, setSelectedId] = useState<string>("");
  const [apiResult, setApiResult] = useState<QAScoreResponse | null>(null);

  const effectiveId = selectedId || reports[0]?.transcript.id || "";
  const selected = useMemo(
    () => reports.find((r) => r.transcript.id === effectiveId) ?? null,
    [reports, effectiveId],
  );

  const pipelineScore = selected?.report.results?.qa?.overall_qa_score ?? null;

  const mutation = useMutation({
    mutationFn: async () => {
      if (!selected) throw new ApiError("Välj ett samtal");
      return apiClient.getQaScore(selected.transcript.segments);
    },
    onSuccess: (data) => setApiResult(data),
    onError: (err) => notifyApiError(err, "QA-score: "),
  });

  const apiScore = apiResult?.qa?.overall_qa_score ?? null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ClipboardCheck className="size-4" />
          QA-scorecard
        </CardTitle>
        <CardDescription>
          Jämför pipeline-inbäddad QA med <code>POST /qa/score</code> för ett valt samtal.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {reports.length === 0 ? (
          <EmptyState
            icon={WifiOff}
            title="Inga samtal"
            hint="Kör demo-pipeline eller ladda upp ett samtal först."
          />
        ) : (
          <>
            <div className="flex flex-wrap items-end gap-2">
              <div className="flex min-w-48 flex-1 flex-col gap-1.5">
                <label htmlFor="qa-call" className="text-xs font-medium text-muted-foreground">
                  Samtal
                </label>
                <select
                  id="qa-call"
                  className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                  value={effectiveId}
                  onChange={(e) => {
                    setSelectedId(e.target.value);
                    setApiResult(null);
                  }}
                >
                  {reports.map((r) => (
                    <option key={r.transcript.id} value={r.transcript.id}>
                      {r.transcript.title} ({r.transcript.id})
                    </option>
                  ))}
                </select>
              </div>
              <Button
                onClick={() => mutation.mutate()}
                disabled={mutation.isPending || !selected}
                className="gap-1.5"
              >
                <ClipboardCheck className="size-4" />
                {mutation.isPending ? "Scorar…" : "Kör /qa/score"}
              </Button>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-border p-3">
                <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
                  Pipeline <Badge variant="outline">analyze_pipeline</Badge>
                </div>
                <p className="text-2xl font-semibold tabular-nums">
                  {pipelineScore == null ? "—" : Math.round(Number(pipelineScore))}
                </p>
              </div>
              <div className="rounded-lg border border-border p-3">
                <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
                  API <Badge variant="outline">/qa/score</Badge>
                </div>
                <p className="text-2xl font-semibold tabular-nums">
                  {apiScore == null ? "—" : Math.round(Number(apiScore))}
                </p>
                {apiResult?.qa?.passed != null ? (
                  <Badge variant={apiResult.qa.passed ? "success" : "warning"} className="mt-2">
                    {apiResult.qa.passed ? "Godkänd" : "Ej godkänd"}
                  </Badge>
                ) : null}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
