"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ListTodo, RefreshCw, Square, WifiOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient, type TranscriptionJobStatus } from "@/lib/api/client";
import { isApiConnected, useHealth } from "@/hooks/use-health";
import { notifyApiError, notifySuccess } from "@/lib/notify";

function statusVariant(status: string): "success" | "warning" | "secondary" | "destructive" {
  const s = status.toLowerCase();
  if (s === "completed" || s === "done") return "success";
  if (s === "running" || s === "pending" || s === "queued") return "warning";
  if (s === "cancelled" || s === "canceled") return "secondary";
  if (s === "failed" || s === "error") return "destructive";
  return "secondary";
}

function canCancel(status: string): boolean {
  const s = status.toLowerCase();
  return s === "running" || s === "pending" || s === "queued";
}

export function TranscriptionJobsPanel() {
  const { data: health } = useHealth();
  const apiOk = isApiConnected(health);
  const qc = useQueryClient();

  const jobsQuery = useQuery({
    queryKey: ["transcription", "jobs"],
    queryFn: () => apiClient.listTranscriptionJobs(30),
    enabled: apiOk,
    refetchInterval: 8_000,
  });

  const cancelMutation = useMutation({
    mutationFn: (jobId: string) => apiClient.cancelTranscriptionJob(jobId),
    onSuccess: (_data, jobId) => {
      notifySuccess(`Avbröt jobb ${jobId.slice(0, 8)}…`);
      void qc.invalidateQueries({ queryKey: ["transcription", "jobs"] });
    },
    onError: (err) => notifyApiError(err, "Avbryt jobb: "),
  });

  const jobs: TranscriptionJobStatus[] = Array.isArray(jobsQuery.data?.jobs)
    ? jobsQuery.data.jobs
    : [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <ListTodo className="size-4" />
              Transkriptionsjobb
            </CardTitle>
            <CardDescription>
              <code>GET /transcription/jobs</code> · avbryt via{" "}
              <code>POST …/cancel</code>
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Uppdatera jobb"
            disabled={!apiOk || jobsQuery.isFetching}
            onClick={() => void jobsQuery.refetch()}
          >
            <RefreshCw className={`size-4 ${jobsQuery.isFetching ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {!apiOk ? (
          <EmptyState
            icon={WifiOff}
            title="API ej ansluten"
            hint="Starta backend och sätt NEXT_PUBLIC_API_KEY om auth krävs."
          />
        ) : jobsQuery.isLoading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : jobsQuery.isError ? (
          <EmptyState
            icon={WifiOff}
            title="Kunde inte hämta jobb"
            hint="Kontrollera att API:et svarar på /transcription/jobs."
          />
        ) : jobs.length === 0 ? (
          <EmptyState
            icon={ListTodo}
            title="Inga jobb ännu"
            hint="Jobb syns här när API:et skapar persistenta transcription jobs."
          />
        ) : (
          <div className="flex flex-col gap-2">
            {jobs.map((job) => (
              <div
                key={job.job_id}
                className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border p-3 text-sm"
              >
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="truncate font-mono text-xs">{job.job_id}</span>
                  <span className="text-xs text-muted-foreground">
                    {job.kind}
                    {job.created_at ? ` · ${String(job.created_at).slice(0, 19)}` : ""}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                  {canCancel(job.status) ? (
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1"
                      disabled={cancelMutation.isPending}
                      onClick={() => cancelMutation.mutate(job.job_id)}
                    >
                      <Square className="size-3" />
                      Avbryt
                    </Button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
