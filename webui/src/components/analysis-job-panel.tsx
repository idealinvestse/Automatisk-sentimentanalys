"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAnalysisJob } from "@/hooks/use-analysis-job";

interface AnalysisJobPanelProps {
  segments: unknown[];
  onCompleted?: (jobId: string) => void;
}

export function AnalysisJobPanel({ segments, onCompleted }: AnalysisJobPanelProps) {
  const { jobId, status, start, cancel, clear } = useAnalysisJob();
  const job = status.data;
  const active = job && ["queued", "running", "cancel_requested"].includes(job.status);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2">
          Lokal LM Studio-analys
          <Badge variant={job?.status === "completed" ? "success" : "secondary"}>
            {job?.status ?? "Inte startad"}
          </Badge>
        </CardTitle>
        <CardDescription>
          Qwen3.5 The Defiant, 70k total kontext, reasoning av. Analysen köas och kan återanslutas efter omladdning.
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
