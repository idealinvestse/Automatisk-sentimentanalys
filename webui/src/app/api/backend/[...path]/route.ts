/**
 * Next.js BFF proxy — keeps SENTIMENT_API_KEY server-side.
 *
 * Browser → /api/backend/<path> → FastAPI with X-API-Key injected.
 * Enable with NEXT_PUBLIC_USE_API_PROXY=1 and set SENTIMENT_API_BASE_URL +
 * SENTIMENT_API_KEY (server-only env vars).
 */

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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

async function proxy(req: NextRequest, pathParts: string[]): Promise<NextResponse> {
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
    init.body = await req.arrayBuffer();
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
