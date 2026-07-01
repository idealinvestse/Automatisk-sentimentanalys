"use client";

import * as React from "react";
import { Radio, Play, Square, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { EmptyState } from "@/components/empty-state";
import { useTranscriptionSocket } from "@/hooks/use-transcription-socket";
import { cn } from "@/lib/utils";
import type { WsConnectionStatus } from "@/lib/transcription-events";

const STATUS_LABEL: Record<WsConnectionStatus, string> = {
  connected: "Ansluten",
  reconnecting: "Återansluter…",
  disconnected: "Frånkopplad",
};

const STATUS_VARIANT: Record<WsConnectionStatus, "success" | "warning" | "secondary"> = {
  connected: "success",
  reconnecting: "warning",
  disconnected: "secondary",
};

const LEVEL_CLASS: Record<string, string> = {
  info: "text-success",
  warning: "text-warning-text",
  error: "text-destructive",
};

export default function TranscriptionPage() {
  const { status, logs, progress, done, connect, disconnect, clearLogs } =
    useTranscriptionSocket();
  const [jobIdInput, setJobIdInput] = React.useState("");
  const logEndRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "end" });
  }, [logs]);

  const isConnected = status !== "disconnected";

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Transkribering</h1>
          <p className="text-sm text-muted-foreground">
            Live-loggar och jobbstatus från backendens WebSocket (<code>/ws/transcription</code>).
          </p>
        </div>
        <Badge variant={STATUS_VARIANT[status]} className="gap-1.5">
          <Radio className="size-3" />
          {STATUS_LABEL[status]}
        </Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Anslutning</CardTitle>
          <CardDescription>
            Prenumerera på ett specifikt jobb-id, eller lämna tomt för alla händelser.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-end gap-3">
          <div className="flex min-w-48 flex-1 flex-col gap-1.5">
            <label htmlFor="job-id" className="text-xs font-medium text-muted-foreground">
              Jobb-id (valfritt)
            </label>
            <Input
              id="job-id"
              placeholder="t.ex. 3f1c2b..."
              value={jobIdInput}
              onChange={(e) => setJobIdInput(e.target.value)}
              disabled={isConnected}
            />
          </div>
          {isConnected ? (
            <Button variant="outline" onClick={disconnect} className="gap-1.5">
              <Square className="size-4" />
              Koppla från
            </Button>
          ) : (
            <Button onClick={() => connect(jobIdInput.trim() || undefined)} className="gap-1.5">
              <Play className="size-4" />
              Anslut
            </Button>
          )}
          <Button variant="ghost" onClick={clearLogs} className="gap-1.5">
            <Trash2 className="size-4" />
            Rensa loggar
          </Button>
        </CardContent>
      </Card>

      {progress ? (
        <Card>
          <CardHeader>
            <CardTitle>Förlopp</CardTitle>
            {progress.current_file ? (
              <CardDescription className="truncate">{progress.current_file}</CardDescription>
            ) : null}
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            <Progress
              value={
                progress.progress !== undefined
                  ? progress.progress * 100
                  : (progress.processed / Math.max(progress.total, 1)) * 100
              }
            />
            <span className="text-xs text-muted-foreground">
              {progress.processed} / {progress.total} filer
            </span>
          </CardContent>
        </Card>
      ) : null}

      {done ? (
        <Card>
          <CardContent className="flex items-center gap-3 pt-5 text-sm">
            <Badge variant={done.failed > 0 ? "warning" : "success"}>Klart</Badge>
            <span>
              {done.ok} lyckades{done.failed > 0 ? `, ${done.failed} misslyckades` : ""}.
            </span>
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Live-loggar</CardTitle>
        </CardHeader>
        <CardContent>
          {logs.length === 0 ? (
            <EmptyState
              icon={Radio}
              title="Inga loggar ännu"
              hint={
                isConnected
                  ? "Väntar på händelser från backend."
                  : "Klicka på Anslut för att börja lyssna på live-transkriberingsloggar."
              }
            />
          ) : (
            <div className="max-h-96 overflow-y-auto rounded-md bg-muted/30 p-3 font-mono text-xs leading-relaxed">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="shrink-0 text-muted-foreground">
                    {log.ts ? log.ts.slice(11, 19) : "--:--:--"}
                  </span>
                  <span className={cn("shrink-0 font-semibold uppercase", LEVEL_CLASS[log.level] ?? "")}>
                    {log.level}
                  </span>
                  <span className="min-w-0 break-words">{log.msg}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          )}
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        Obs: webbläsarens WebSocket-API kan inte skicka <code>X-API-Key</code>-headern, så denna
        vy fungerar just nu bara mot en backend utan API-nyckel/auth (se
        docs/WEBUI_MODERNIZATION_PLAN.md, Fas 3).
      </p>
    </div>
  );
}
