"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAnalysisJob } from "@/hooks/use-analysis-job";
import { isDirectApiEnabled } from "@/lib/api/client";

interface AnalysisJobPanelProps {
  segments: unknown[];
  onCompleted?: (jobId: string) => void;
}

export function AnalysisJobPanel({ segments, onCompleted }: AnalysisJobPanelProps) {
  const labEnabled = isDirectApiEnabled();
  const { jobId, status, start, cancel, clear } = useAnalysisJob();
  const job = status.data;
  const active = job && ["queued", "running", "cancel_requested"].includes(job.status);

  if (!labEnabled) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Lokal LM Studio-analys</CardTitle>
          <CardDescription>
            Labbläge — inte kundpilot. BFF-proxyn tillåter inte <code>/analysis/jobs</code>.
            Sätt <code>NEXT_PUBLIC_USE_DIRECT_API=1</code> bara på en isolerad labbmaskin.
            Referensmodellen (uncensored) är avvisad för kunddata.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2">
          Lokal LM Studio-analys (labb)
          <Badge variant={job?.status === "completed" ? "success" : "secondary"}>
            {job?.status ?? "Inte startad"}
          </Badge>
        </CardTitle>
        <CardDescription>
          Ej pilot-godkänd. Qwen-uncensored är avvisad för kunddata. Direct API + loopback
          LM Studio, 70k kontext. Köas och kan återanslutas efter omladdning.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center gap-3">
        <Button
          disabled={segments.length === 0 || Boolean(active) || start.isPending}
          onClick={() => start.mutate(segments)}
        >
          Starta lokal analys
        </Button>
        {active && (
          <Button variant="outline" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
            Avbryt
          </Button>
        )}
        {job?.result_available && jobId && (
          <Button variant="outline" onClick={() => onCompleted?.(jobId)}>
            Hämta rapport
          </Button>
        )}
        {job && !active && (
          <Button variant="ghost" onClick={clear}>
            Rensa status
          </Button>
        )}
        {job && (
          <span className="text-sm text-muted-foreground">
            Fas: {job.phase}
            {job.error_code ? ` · ${job.error_code}` : ""}
          </span>
        )}
      </CardContent>
    </Card>
  );
}
