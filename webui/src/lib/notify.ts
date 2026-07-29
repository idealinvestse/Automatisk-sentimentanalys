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
  if (err instanceof ApiError && err.status === 401) {
    msg =
      `${prefix}Otillåten (401) — kontrollera NEXT_PUBLIC_API_KEY / SENTIMENT_API_KEY`.trim();
  } else if (err instanceof ApiError) {
    const detail = formatApiDetail(err.detail);
    const base = detail ? `${err.message}: ${detail}` : err.message;
    msg = err.status
      ? `${prefix}${base} (HTTP ${err.status})`.trim()
      : `${prefix}${base}`.trim();
  } else if (prefix) {
    msg = `${prefix}${msg}`;
  }
  notifyError(msg);
}

/** Flatten FastAPI / custom API error payloads for toasts. */
function formatApiDetail(detail: unknown): string {
  if (detail == null) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          const loc = Array.isArray((item as { loc?: unknown }).loc)
            ? (item as { loc: unknown[] }).loc.join(".")
            : "";
          const msg = String((item as { msg: unknown }).msg);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return String(item);
      })
      .filter(Boolean)
      .join("; ");
  }
  if (typeof detail === "object" && detail !== null && "detail" in detail) {
    return formatApiDetail((detail as { detail: unknown }).detail);
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}
