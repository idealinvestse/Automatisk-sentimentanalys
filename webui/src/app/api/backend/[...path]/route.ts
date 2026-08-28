/**
 * Next.js BFF proxy — keeps SENTIMENT_API_KEY server-side.
 *
 * Browser → /api/backend/<path> → FastAPI with X-API-Key injected.
 * Default in the webui client; disable with NEXT_PUBLIC_USE_DIRECT_API=1.
 */

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Browser-facing FastAPI paths.
 *
 * This is deliberately an exact allowlist instead of a router-prefix list:
 * the BFF adds a server-side credential, so exposing an extra admin or
 * filesystem-oriented endpoint would otherwise amplify browser privileges.
 */
const ALLOWED_PATHS = new Set([
  "health",
  "ready",
  "ws/transcription/ticket",
  "analyze_pipeline",
  "analyze_pipeline/partial",
  "analyze_pipeline/compare",
  "agent_performance/:id",
  "insights/hot_topics",
  "search/semantic",
  "qa/score",
  "alerts",
  "alerting/status",
  "alerting/reset-circuit-breaker",
  "calls",
  "calls/:id",
  "llm/analysis-profiles",
  "llm/analysis-profiles/:id",
  "upload",
  "transcribe",
  "transcription/jobs",
  "transcription/jobs/:id",
  "transcription/jobs/:id/cancel",
  "edge/analyze-text",
  "edge/analyze-segments",
]);

function maxBodyBytes(): number {
  // Aligns with FastAPI default of 200MB (max_upload_size_mb = 200)
  const mb = Number(process.env.API_MAX_UPLOAD_SIZE_MB || "200");
  return Math.max(1, Number.isFinite(mb) ? mb : 200) * 1024 * 1024;
}

function upstreamBase(): string {
  const base =
    process.env.SENTIMENT_API_BASE_URL?.trim() ||
    process.env.API_BASE_URL?.trim() ||
    "http://localhost:8000";
  return base.replace(/\/$/, "");
}

function apiKey(): string | undefined {
  // The BFF credential is authoritative. Never let a browser-supplied key
  // override it, or an obsolete public key can turn a working BFF into 401s.
  return process.env.SENTIMENT_API_KEY?.trim() || undefined;
}

function isAllowedPath(parts: string[]): boolean {
  const path = parts.map((part) => part.toLowerCase()).join("/");
  const normalized = path
    .split("/")
    .map((part, index) => {
      if (
        (index === 1 && ["agent_performance", "calls"].includes(parts[0]?.toLowerCase() ?? "")) ||
        (index === 2 && parts[0]?.toLowerCase() === "llm") ||
        (index === 2 && parts[0]?.toLowerCase() === "transcription")
      ) {
        return ":id";
      }
      return part;
    })
    .join("/");
  return ALLOWED_PATHS.has(normalized);
}

async function proxy(req: NextRequest, pathParts: string[]): Promise<NextResponse> {
  if (!isAllowedPath(pathParts)) {
    return NextResponse.json({ detail: "Path not allowed via BFF proxy" }, { status: 404 });
  }

  const targetPath = pathParts.map(encodeURIComponent).join("/");
  const url = new URL(req.url);
  const upstream = `${upstreamBase()}/${targetPath}${url.search}`;

  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const requestId = req.headers.get("x-request-id");
  if (requestId) headers.set("x-request-id", requestId);
  const authHeader = req.headers.get("authorization");
  if (authHeader) headers.set("authorization", authHeader);
  const key = apiKey();
  if (key) headers.set("x-api-key", key);

  // The client permits model comparison for nine minutes and ASR for ten.
  // Keep the server-side boundary at least as long so the BFF path is not a
  // shorter, surprising deadline.
  const controller = new AbortController();
  const timeoutMs = 600_000;
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const init: RequestInit = {
    method: req.method,
    headers,
    redirect: "manual",
    signal: controller.signal,
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    const buf = await req.arrayBuffer();
    if (buf.byteLength > maxBodyBytes()) {
      clearTimeout(timeoutId);
      return NextResponse.json(
        { detail: `Payload too large (max ${Math.round(maxBodyBytes() / (1024 * 1024))} MB)` },
        { status: 413 },
      );
    }
    init.body = buf;
  }

  let upstreamRes: Response;
  try {
    upstreamRes = await fetch(upstream, init);
  } catch (err) {
    clearTimeout(timeoutId);
    const isTimeout = err instanceof Error && err.name === "AbortError";
    return NextResponse.json(
      {
        detail: isTimeout
          ? `Upstream API request timed out (${timeoutMs}ms)`
          : `Upstream API unreachable: ${String(err)}`,
      },
      { status: isTimeout ? 504 : 502 },
    );
  } finally {
    clearTimeout(timeoutId);
  }

  const outHeaders = new Headers();
  const ct = upstreamRes.headers.get("content-type");
  if (ct) outHeaders.set("content-type", ct);
  const rid = upstreamRes.headers.get("x-request-id");
  if (rid) outHeaders.set("x-request-id", rid);

  const body = await upstreamRes.arrayBuffer();
  return new NextResponse(body, {
    status: upstreamRes.status,
    headers: outHeaders,
  });
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function POST(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function PUT(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function DELETE(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function PATCH(req: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(req, path);
}
