"use client";

import * as React from "react";

import { ApiError, apiClient } from "@/lib/api/client";
import type {
  DoneEvent,
  LogEvent,
  PartialAnalysisEvent,
  ProgressEvent,
  StatusEvent,
  TranscriptionEvent,
  WsConnectionStatus,
} from "@/lib/transcription-events";

const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 30_000;
const MAX_ATTEMPTS = 8;
const MAX_LOGS = 300;
const PING_INTERVAL_MS = 25_000;

function backoffDelay(attempt: number): number {
  const exp = Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** attempt);
  const jitter = Math.floor(Math.random() * 400);
  return exp + jitter;
}

/**
 * Client for WS /ws/transcription with reconnect/backoff.
 *
 * Auth: browsers cannot set X-API-Key on the WS handshake — `apiClient.wsUrl()`
 * fetches GET /ws/transcription/ticket (with API key) and appends `?token=`.
 */
export function useTranscriptionSocket() {
  const [status, setStatus] = React.useState<WsConnectionStatus>("disconnected");
  const [logs, setLogs] = React.useState<LogEvent[]>([]);
  const [progress, setProgress] = React.useState<ProgressEvent | null>(null);
  const [done, setDone] = React.useState<DoneEvent | null>(null);
  const [partialAnalysis, setPartialAnalysis] = React.useState<PartialAnalysisEvent | null>(null);
  const [jobId, setJobId] = React.useState<string | null>(null);

  const wsRef = React.useRef<WebSocket | null>(null);
  const attemptRef = React.useRef(0);
  const authRetryRef = React.useRef(0);
  const stoppedRef = React.useRef(true);
  const timeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  const clearRetryTimer = React.useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const clearPing = React.useCallback(() => {
    if (pingRef.current) {
      clearInterval(pingRef.current);
      pingRef.current = null;
    }
  }, []);

  const closeSocket = React.useCallback(() => {
    clearPing();
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [clearPing]);

  const handleEvent = React.useCallback((event: TranscriptionEvent) => {
    if (event.type === "log") {
      setLogs((prev) => [...prev.slice(-(MAX_LOGS - 1)), event]);
    } else if (event.type === "progress") {
      setProgress(event);
    } else if (event.type === "done") {
      setDone(event);
    } else if (event.type === "partial_analysis") {
      setPartialAnalysis(event);
      setLogs((prev) => [
        ...prev.slice(-(MAX_LOGS - 1)),
        {
          type: "log",
          job_id: event.job_id,
          level: "info",
          msg: `Delanalys: ${event.segment_count} segment, ${event.sentiment_count} sentiment`,
          ts: event.ts,
        },
      ]);
    } else if (event.type === "status") {
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
    if (attemptRef.current >= MAX_ATTEMPTS) {
      setStatus("disconnected");
      return;
    }
    setStatus("reconnecting");
    const delay = backoffDelay(attemptRef.current);
    attemptRef.current += 1;
    timeoutRef.current = setTimeout(() => connectRef.current(targetJobId), delay);
  }, []);

  React.useEffect(() => {
    connectRef.current = async (targetJobId: string | null) => {
      if (stoppedRef.current) return;
      clearRetryTimer();
      closeSocket();
      setStatus(attemptRef.current > 0 ? "reconnecting" : "disconnected");

      let ws: WebSocket;
      try {
        const url = await apiClient.wsUrl();
        ws = new WebSocket(url);
      } catch (err) {
        if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
          stoppedRef.current = true;
          setStatus("unauthorized");
          return;
        }
        scheduleRetry(targetJobId);
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
        authRetryRef.current = 0;
        setStatus("connected");
        clearPing();
        pingRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, PING_INTERVAL_MS);
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
      ws.onclose = (ev) => {
        wsRef.current = null;
        clearPing();
        if (ev.code === 1008) {
          if (authRetryRef.current === 0) {
            authRetryRef.current = 1;
            scheduleRetry(targetJobId);
            return;
          }
          stoppedRef.current = true;
          setStatus("unauthorized");
          return;
        }
        if (!stoppedRef.current) scheduleRetry(targetJobId);
        else setStatus("disconnected");
      };
      ws.onerror = () => {
        ws.close();
      };
    };
  }, [clearPing, clearRetryTimer, closeSocket, handleEvent, scheduleRetry]);

  const connect = React.useCallback((targetJobId?: string) => {
    stoppedRef.current = false;
    attemptRef.current = 0;
    authRetryRef.current = 0;
    setJobId(targetJobId ?? null);
    setDone(null);
    setPartialAnalysis(null);
    connectRef.current(targetJobId ?? null);
  }, []);

  const disconnect = React.useCallback(() => {
    stoppedRef.current = true;
    clearRetryTimer();
    closeSocket();
    setStatus("disconnected");
  }, [clearRetryTimer, closeSocket]);

  const clearLogs = React.useCallback(() => setLogs([]), []);

  React.useEffect(() => {
    return () => {
      stoppedRef.current = true;
      clearRetryTimer();
      closeSocket();
    };
  }, [clearRetryTimer, closeSocket]);

  return {
    status,
    logs,
    progress,
    done,
    partialAnalysis,
    jobId,
    connect,
    disconnect,
    clearLogs,
  };
}
