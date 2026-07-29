/**
 * Event shapes for /ws/transcription, mirroring src/api/transcription_events.py
 * and partial_analysis emits from routers/transcription.py.
 */

export interface BaseEvent {
  type: string;
  job_id?: string | null;
  ts?: string;
}

export interface ConnectedEvent extends BaseEvent {
  type: "connected";
  msg: string;
}

export interface SubscribedEvent extends BaseEvent {
  type: "subscribed";
  job_id: string | null;
}

export interface LogEvent extends BaseEvent {
  type: "log";
  level: "info" | "warning" | "error" | string;
  msg: string;
  file?: string | null;
}

export interface ProgressEvent extends BaseEvent {
  type: "progress";
  processed: number;
  total: number;
  current_file?: string | null;
  progress?: number;
}

export interface StatusEvent extends BaseEvent {
  type: "status";
  is_running: boolean;
  [key: string]: unknown;
}

export interface DoneEvent extends BaseEvent {
  type: "done";
  ok: number;
  failed: number;
}

export interface PongEvent extends BaseEvent {
  type: "pong";
}

/** Emitted when run_partial_analysis=true after ASR (local incremental path). */
export interface PartialAnalysisEvent extends BaseEvent {
  type: "partial_analysis";
  job_id: string | null;
  segment_count: number;
  sentiment_count: number;
}

export type TranscriptionEvent =
  | ConnectedEvent
  | SubscribedEvent
  | LogEvent
  | ProgressEvent
  | StatusEvent
  | DoneEvent
  | PongEvent
  | PartialAnalysisEvent;

export type WsConnectionStatus = "connected" | "reconnecting" | "disconnected";
