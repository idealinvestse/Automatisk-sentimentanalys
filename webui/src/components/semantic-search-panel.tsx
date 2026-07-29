"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Search, WifiOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { EmptyState } from "@/components/empty-state";
import { apiClient, ApiError, type SemanticSearchResponse } from "@/lib/api/client";
import { notifyApiError } from "@/lib/notify";
import type { RealCall } from "@/lib/real-data";

export function SemanticSearchPanel({ reports }: { reports: RealCall[] }) {
  const [query, setQuery] = useState("faktura");
  const [result, setResult] = useState<SemanticSearchResponse | null>(null);

  const mutation = useMutation({
    mutationFn: async () => {
      const q = query.trim();
      if (!q) throw new ApiError("Ange en sökfråga");
      if (reports.length === 0) throw new ApiError("Inga samtal att söka i — ladda demo eller live-data först");
      const segmentsList = reports.map((r) => r.transcript.segments);
      return apiClient.semanticSearch(q, segmentsList, { top_k: 8 });
    },
    onSuccess: (data) => setResult(data),
    onError: (err) => notifyApiError(err, "Semantisk sökning: "),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="size-4" />
          Semantisk sökning
        </CardTitle>
        <CardDescription>
          <code>POST /search/semantic</code> över aktuella samtal (demo eller live).
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex min-w-48 flex-1 flex-col gap-1.5">
            <label htmlFor="semantic-q" className="text-xs font-medium text-muted-foreground">
              Fråga
            </label>
            <Input
              id="semantic-q"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") mutation.mutate();
              }}
              placeholder="t.ex. felaktig faktura"
              disabled={mutation.isPending}
            />
          </div>
          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || reports.length === 0}
            className="gap-1.5"
          >
            <Search className="size-4" />
            {mutation.isPending ? "Söker…" : "Sök"}
          </Button>
        </div>

        {reports.length === 0 ? (
          <EmptyState
            icon={WifiOff}
            title="Inga samtal att indexera"
            hint="Vänta på pipeline-demo eller ladda upp ett samtal under Transkribering."
          />
        ) : null}

        {result ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>
                {result.hits.length} träffar för &quot;{result.query}&quot;
              </span>
              <Badge variant="outline">API</Badge>
            </div>
            {result.hits.length === 0 ? (
              <p className="text-sm text-muted-foreground">Inga träffar.</p>
            ) : (
              result.hits.map((hit, i) => (
                <div key={i} className="rounded-lg border border-border p-3 text-sm">
                  <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    {hit.score != null ? (
                      <Badge variant="secondary">score {Number(hit.score).toFixed(3)}</Badge>
                    ) : null}
                    {hit.speaker ? <span>{String(hit.speaker)}</span> : null}
                    {hit.call_index != null ? <span>call #{hit.call_index}</span> : null}
                  </div>
                  <p className="leading-relaxed">{String(hit.text ?? hit.snippet ?? JSON.stringify(hit))}</p>
                </div>
              ))
            )}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
