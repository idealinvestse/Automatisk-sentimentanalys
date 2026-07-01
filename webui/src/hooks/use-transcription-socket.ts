"use client";

import * as React from "react";

import { apiClient } from "@/lib/api/client";
import type {
  DoneEvent,
  LogEvent,
  ProgressEvent,
  StatusEvent,
  TranscriptionEvent,
  WsConnectionStatus,
} from "@/lib/transcription-events";

const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 30_000;
const MAX_LOGS = 300;

/**
 * Client for GET /ws/transcription, mirroring the reconnect/backoff behavior
 * of app/nicegui_dashboard/services/transcription_ws_client.py.
 *
 * Note: unlike the Python client, browsers cannot set custom headers (e.g.
 * X-API-Key) on a WebSocket handshake, so this only works out of the box
 * when SENTIMENT_API_KEY / auth is disabled on the backend. See
 * docs/WEBUI_MODERNIZATION_PLAN.md for the Fas 3 follow-up on WS auth.
 */
export function useTranscriptionSocket() {
  const [status, setStatus] = React.useState<WsConnectionStatus>("disconnected");
  const [logs, setLogs] = React.useState<LogEvent[]>([]);
  const [progress, setProgress] = React.useState<ProgressEvent | null>(null);
  const [done, setDone] = React.useState<DoneEvent | null>(null);
  const [jobId, setJobId] = React.useState<string | null>(null);

  const wsRef = React.useRef<WebSocket | null>(null);
  const attemptRef = React.useRef(0);
  const stoppedRef = React.useRef(true);
  const timeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearRetryTimer = React.useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const handleEvent = React.useCallback((event: TranscriptionEvent) => {
    if (event.type === "log") {
      setLogs((prev) => [...prev.slice(-(MAX_LOGS - 1)), event]);
    } else if (event.type === "progress") {
      setProgress(event);
    } else if (event.type === "done") {
      setDone(event);
    } else if (event.type === "status") {
      // Surface long-running-state changes as a synthetic log line so they
      // are visible in the same feed without a separate status widget.
      const statusEvent = event as StatusEvent;
      setLogs((prev) => [
        ...prev.slice(-(MAX_LOGS - 1)),
        {
          type: "log",
          job_id: statusEvent.job_id,
          level: "info",
          msg: statusEvent.is_running ? "Jobb körs..." : "Jobb pausat/klart.",
          ts: statusEvent.ts,
        },
      ]);
    }
  }, []);

  const connectRef = React.useRef<(jobId: string | null) => void>(() => {});

  const scheduleRetry = React.useCallback((targetJobId: string | null) => {
    if (stoppedRef.current) return;
    setStatus("reconnecting");
    const delay = Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** attemptRef.current);
    attemptRef.current += 1;
    timeoutRef.current = setTimeout(() => connectRef.current(targetJobId), delay);
  }, []);

  // Keep the connect closure in a ref so callers (connect/scheduleRetry) always
  // invoke the latest version without re-creating their own callbacks.
  React.useEffect(() => {
    connectRef.current = (targetJobId: string | null) => {
      if (stoppedRef.current) return;
      clearRetryTimer();
      setStatus(attemptRef.current > 0 ? "reconnecting" : "disconnected");

      let ws: WebSocket;
      try {
        ws = new WebSocket(apiClient.wsUrl());
      } catch {
        scheduleRetry(targetJobId);
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
        setStatus("connected");
        if (targetJobId) {
          ws.send(JSON.stringify({ type: "subscribe", job_id: targetJobId }));
        }
      };
      ws.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data) as TranscriptionEvent;
          handleEvent(parsed);
        } catch {
          // ignore malformed frames
        }
      };
      ws.onclose = () => {
        wsRef.current = null;
        if (!stoppedRef.current) scheduleRetry(targetJobId);
        else setStatus("disconnected");
      };
      ws.onerror = () => {
        ws.close();
      };
    };
  }, [clearRetryTimer, handleEvent, scheduleRetry]);

  const connect = React.useCallback((targetJobId?: string) => {
    stoppedRef.current = false;
    attemptRef.current = 0;
    setJobId(targetJobId ?? null);
    setDone(null);
    connectRef.current(targetJobId ?? null);
  }, []);

  const disconnect = React.useCallback(() => {
    stoppedRef.current = true;
    clearRetryTimer();
    wsRef.current?.close();
    wsRef.current = null;
    setStatus("disconnected");
  }, [clearRetryTimer]);

  const clearLogs = React.useCallback(() => setLogs([]), []);

  React.useEffect(() => {
    return () => {
      stoppedRef.current = true;
      clearRetryTimer();
      wsRef.current?.close();
    };
  }, [clearRetryTimer]);

  return { status, logs, progress, done, jobId, connect, disconnect, clearLogs };
}
