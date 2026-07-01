"use client";

import { toast } from "sonner";

import { ApiError } from "@/lib/api/client";

/**
 * Toast helpers mirroring app/nicegui_dashboard/services/ui_helpers.py:
 * notify_success / notify_warning / notify_error / notify_api_error.
 */
export function notifySuccess(message: string) {
  toast.success(message);
}

export function notifyWarning(message: string) {
  toast.warning(message);
}

export function notifyError(message: string) {
  toast.error(message);
}

export function notifyApiError(err: unknown, prefix = "") {
  let msg = err instanceof Error ? err.message : String(err);
  if (err instanceof ApiError && err.status) {
    msg = `${prefix}${msg} (HTTP ${err.status})`.trim();
  } else if (prefix) {
    msg = `${prefix}${msg}`;
  }
  notifyError(msg);
}
