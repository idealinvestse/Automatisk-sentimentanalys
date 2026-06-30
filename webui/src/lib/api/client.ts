/**
 * Typed fetch client for the FastAPI backend.
 *
 * Mirrors app/nicegui_dashboard/services/nicegui_api_client.py so the new
 * web UI talks to the exact same REST endpoints as the legacy NiceGUI
 * dashboard. No backend changes are required to use this client.
 */

const DEFAULT_BASE_URL = "http://localhost:8000";

export class ApiError extends Error {
  status?: number;
  detail?: unknown;

  constructor(message: string, status?: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export interface ApiClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
}

function getBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_BASE_URL;
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? getBaseUrl()).replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  private headers(): HeadersInit {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers["X-API-Key"] = this.apiKey;
    return headers;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: { ...this.headers(), ...(init.headers ?? {}) },
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new ApiError(`Timeout mot ${path} (${this.timeoutMs}ms)`);
      }
      throw new ApiError(`Kan inte ansluta till backend (${this.baseUrl}): ${String(err)}`);
    } finally {
      clearTimeout(timeout);
    }

    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = await response.text();
      }
      throw new ApiError(`API-fel ${response.status} på ${path}`, response.status, detail);
    }
    return (await response.json()) as T;
  }

  get<T>(path: string, params?: Record<string, string | number | boolean | undefined>) {
    const search = params
      ? "?" +
        new URLSearchParams(
          Object.entries(params)
            .filter(([, v]) => v !== undefined)
            .map(([k, v]) => [k, String(v)]),
        ).toString()
      : "";
    return this.request<T>(`${path}${search}`, { method: "GET" });
  }

  post<T>(path: string, body: unknown) {
    return this.request<T>(path, { method: "POST", body: JSON.stringify(body) });
  }

  async health(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(10_000) });
      return response.ok;
    } catch {
      return false;
    }
  }

  analyzePipeline(segments: unknown[], options: Record<string, unknown> = {}) {
    return this.post("/analyze_pipeline", { segments, profile: "callcenter", ...options });
  }

  getAlertingStatus<T = unknown>() {
    return this.get<T>("/alerting/status");
  }

  getProcessEvents<T = unknown>(params: { limit?: number; job_id?: string; component?: string; level?: string } = {}) {
    return this.get<T>("/status/processes", { limit: params.limit ?? 100, ...params });
  }

  getJobStatus<T = unknown>(jobId: string) {
    return this.get<T>(`/status/jobs/${jobId}`);
  }
}

export const apiClient = new ApiClient();
