/**
 * Next.js BFF proxy — keeps SENTIMENT_API_KEY server-side.
 *
 * Browser → /api/backend/<path> → FastAPI with X-API-Key injected.
 * Default in the webui client; disable with NEXT_PUBLIC_USE_DIRECT_API=1.
 */

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ALLOWED_PREFIXES = [
  "health",
  "ready",
  "metrics",
  "status",
  "analyze",
  "analyze_pipeline",
  "analyze_conversation",
  "agent_performance",
  "insights",
  "search",
  "qa",
  "alerts",
  "alerting",
  "calls",
  "llm",
  "upload",
  "transcribe",
  "batch_transcribe",
  "batch_analyze_conversation",
  "scan_process",
  "transcription",
  "ws",
  "edge",
] as const;

function maxBodyBytes(): number {
  const mb = Number(process.env.API_MAX_UPLOAD_SIZE_MB || "50");
  return Math.max(1, Number.isFinite(mb) ? mb : 50) * 1024 * 1024;
}

function upstreamBase(): string {
  const base =
    process.env.SENTIMENT_API_BASE_URL?.trim() ||
    process.env.API_BASE_URL?.trim() ||
    "http://localhost:8000";
  return base.replace(/\/$/, "");
}

function apiKey(): string | undefined {
  return process.env.SENTIMENT_API_KEY?.trim() || undefined;
}

function isAllowedPath(parts: string[]): boolean {
  const first = (parts[0] ?? "").toLowerCase();
  return (ALLOWED_PREFIXES as readonly string[]).includes(first);
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
  const key = apiKey();
  if (key) headers.set("x-api-key", key);

  const init: RequestInit = {
    method: req.method,
    headers,
    redirect: "manual",
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    const buf = await req.arrayBuffer();
    if (buf.byteLength > maxBodyBytes()) {
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
    return NextResponse.json(
      { detail: `Upstream API unreachable: ${String(err)}` },
      { status: 502 },
    );
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
