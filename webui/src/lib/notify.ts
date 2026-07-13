"use client";

import { toast } from "sonner";

import { ApiError } from "@/lib/api/client";

/**
 * Toast helpers: success / warning / error / API error.
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
