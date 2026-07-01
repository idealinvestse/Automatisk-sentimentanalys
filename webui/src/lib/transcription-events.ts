/**
 * Event shapes for /ws/transcription, mirroring src/api/transcription_events.py
 * (TranscriptionEventHub.log/progress/status/done) and the reference client
 * app/nicegui_dashboard/services/transcription_ws_client.py.
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

export type TranscriptionEvent =
  | ConnectedEvent
  | SubscribedEvent
  | LogEvent
  | ProgressEvent
  | StatusEvent
  | DoneEvent
  | PongEvent;

export type WsConnectionStatus = "connected" | "reconnecting" | "disconnected";
