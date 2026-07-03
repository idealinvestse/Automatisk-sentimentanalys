"use client";

import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { Cpu, WifiOff, Info, Zap, FileText, Layers } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { EmptyState } from "@/components/empty-state";
import { SentimentBadge } from "@/components/status-badges";
import { apiClient, ApiError, type EdgeAnalysisResult } from "@/lib/api/client";
import { notifyApiError, notifySuccess } from "@/lib/notify";
import { useHealth } from "@/hooks/use-health";

type Mode = "text" | "segments";

const EXAMPLE_TEXT = "Tack för hjälpen, det fungerade bra! Ni var väldigt professionella.";

const EXAMPLE_SEGMENTS = JSON.stringify(
  [
    { text: "Hej, jag ringer angående min faktura.", speaker: "Kund" },
    { text: "Hej! Jag hjälper dig gärna med det. Vad är problemet?", speaker: "Agent" },
    { text: "Jag har blivit debiterad dubbelt och det är frustrerande.", speaker: "Kund" },
  ],
  null,
  2,
);

export default function EdgePage() {
  const { data: connected } = useHealth();
  const [mode, setMode] = React.useState<Mode>("text");
  const [textInput, setTextInput] = React.useState("");
  const [segmentsInput, setSegmentsInput] = React.useState("");

  const mutation = useMutation<EdgeAnalysisResult, ApiError, void>({
    mutationFn: async () => {
      if (mode === "text") {
        const text = textInput.trim();
        if (!text) throw new ApiError("Ange text att analysera");
        return apiClient.edgeAnalyzeText(text);
      }
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
      return apiClient.edgeAnalyzeSegments(
        segments as { text: string; speaker?: string }[],
      );
    },
    onSuccess: () => notifySuccess("Edge-analys klar"),
    onError: (err) => notifyApiError(err, "Edge-fel: "),
  });

  const result = mutation.data;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Edge AI</h1>
          <p className="text-sm text-muted-foreground">
            Offline sentiment + intent — motsvarar <code>sentimentanalys edge-analyze</code>.
          </p>
        </div>
        <Badge variant={connected ? "success" : "warning"}>
          {connected ? "API ansluten" : "API ej verifierad"}
        </Badge>
      </div>

      {/* Info banner */}
      <div className="flex items-start gap-2 rounded-md border border-blue-500/30 bg-blue-500/5 p-3 text-xs">
        <Info className="mt-0.5 size-3.5 shrink-0 text-blue-500" />
        <span>
          Edge AI kör <strong>endast</strong> lokal sentimentanalys (XLM-RoBERTa) + heuristisk
          intent. Inget LLM, ingen diarization, inga Fas 4-aggregat. PII-redigering appliceras
          för <code>callcenter</code>-profilen. Designat för offline/edge-deployment.
        </span>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2">
        <Button
          variant={mode === "text" ? "default" : "outline"}
          size="sm"
          onClick={() => setMode("text")}
          className="gap-1.5"
        >
          <FileText className="size-4" />
          Enkel text
        </Button>
        <Button
          variant={mode === "segments" ? "default" : "outline"}
          size="sm"
          onClick={() => setMode("segments")}
          className="gap-1.5"
        >
          <Layers className="size-4" />
          Segment-lista (JSON)
        </Button>
      </div>

      {/* Input card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="size-4" />
            {mode === "text" ? "Textanalys (offline)" : "Segmentanalys (offline)"}
          </CardTitle>
          <CardDescription>
            {mode === "text"
              ? "Ange en textrad för sentiment + intent."
              : "Klistra in pre-transkriberade segment som JSON."}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {mode === "text" ? (
            <Textarea
              rows={3}
              placeholder={EXAMPLE_TEXT}
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
            />
          ) : (
            <Textarea
              rows={8}
              placeholder={EXAMPLE_SEGMENTS}
              value={segmentsInput}
              onChange={(e) => setSegmentsInput(e.target.value)}
              className="font-mono text-xs"
            />
          )}

          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="w-fit gap-1.5"
          >
            <Zap className="size-4" />
            {mutation.isPending ? "Analyserar…" : "Kör edge-analys"}
          </Button>
        </CardContent>
      </Card>

      {/* Result card */}
      <Card>
        <CardHeader>
          <CardTitle>Resultat</CardTitle>
        </CardHeader>
        <CardContent>
          {mutation.isIdle ? (
            <EmptyState
              icon={Cpu}
              title="Inga resultat ännu"
              hint={mode === "text" ? "Ange text och kör analys." : "Klistra in segments och kör analys."}
            />
          ) : mutation.isPending ? (
            <p className="text-sm text-muted-foreground">Kör offline-analys…</p>
          ) : mutation.isError ? (
            <div className="flex flex-col gap-2">
              <Badge variant="destructive" className="w-fit">API-fel</Badge>
              <p className="text-sm">{mutation.error.message}</p>
            </div>
          ) : result ? (
            <EdgeResultView result={result} />
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}

function EdgeResultView({ result }: { result: EdgeAnalysisResult }) {
  return (
    <div className="flex flex-col gap-4">
      {/* Meta badges */}
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="success" className="gap-1">
          <WifiOff className="size-3" />
          Offline
        </Badge>
        <Badge variant="secondary">Profil: {result.profile}</Badge>
        <Badge variant="outline">LLM: {result.llm_used ? "ja" : "nej"}</Badge>
        <Badge variant="outline">{result.segments.length} segment</Badge>
      </div>

      {/* Summary */}
      {result.summary ? (
        <div className="rounded-md bg-muted/50 p-3 text-sm">{result.summary}</div>
      ) : null}

      {/* Segment results */}
      <div className="flex flex-col gap-2">
        {result.segments.map((seg, i) => (
          <div
            key={i}
            className="flex flex-col gap-1.5 rounded-lg border border-border p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-sm">{seg.text}</span>
              <div className="flex shrink-0 items-center gap-1.5">
                {seg.sentiment_label ? (
                  <SentimentBadge
                    value={
                      seg.sentiment_label.includes("pos")
                        ? "positive"
                        : seg.sentiment_label.includes("neg")
                          ? "negative"
                          : "neutral"
                    }
                  />
                ) : null}
                {seg.sentiment_score !== null ? (
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {seg.sentiment_score.toFixed(2)}
                  </span>
                ) : null}
              </div>
            </div>
            {seg.intent ? (
              <Badge variant="outline" className="w-fit text-xs capitalize">
                Intent: {seg.intent.replace(/_/g, " ")}
              </Badge>
            ) : null}
          </div>
        ))}
      </div>

      {/* Limitations */}
      {result.limitations.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Begränsningar</span>
          <div className="flex flex-wrap gap-1.5">
            {result.limitations.map((lim) => (
              <Badge key={lim} variant="warning" className="text-xs">
                {lim}
              </Badge>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
