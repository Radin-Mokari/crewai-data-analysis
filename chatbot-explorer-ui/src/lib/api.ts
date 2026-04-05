/**
 * FastAPI backend (server.py). Default: same-origin `/api/*` via Vite dev proxy → http://127.0.0.1:8765.
 * Override with VITE_API_BASE_URL (no trailing slash), e.g. http://127.0.0.1:8765
 */

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export type HealthOk = {
  status: "ok";
  run_id: string;
  run_dir: string;
};

export type HealthNoWorkflow = {
  status: "no_workflow";
  hint: string;
};

export type HealthResponse = HealthOk | HealthNoWorkflow;

export type SpecialistStep = {
  agent: string;
  excerpt: string;
};

export type ChatResponse = {
  outcome: string;
  lines: string[];
  run_id: string;
  specialist_steps?: SpecialistStep[];
  /** Server paths like `/artifacts/{run_id}/charts/file.png` — use `resolveArtifactUrl` for img src. */
  chart_urls?: string[];
};

export type ReportResponse = {
  ok: boolean;
  run_dir: string;
  report_markdown?: string;
  truncated?: boolean;
};

export type ResetResponse = {
  ok: boolean;
  run_id: string;
  run_dir: string;
};

/** POST /pipeline — full dynamic supervisor batch until DONE / guardrails / DYNAMIC_MAX_STEPS */
export type PipelineResponse = {
  ok: boolean;
  run_id: string;
  specialist_steps: number;
};

function apiBase(): string {
  const v = import.meta.env.VITE_API_BASE_URL;
  if (v != null && String(v).trim() !== "") {
    return String(v).replace(/\/$/, "");
  }
  return "/api";
}

/** Prefix an artifact path (`/artifacts/...` from the API) for `<img src>` or fetches. Works with Vite `/api` proxy or `VITE_API_BASE_URL`. */
export function resolveArtifactUrl(artifactPath: string): string {
  const base = apiBase().replace(/\/$/, "");
  const p = artifactPath.startsWith("/") ? artifactPath : `/${artifactPath}`;
  return `${base}${p}`;
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const j: unknown = await res.json();
    if (j && typeof j === "object" && "detail" in j) {
      const d = (j as { detail: unknown }).detail;
      if (typeof d === "string") return d;
      if (Array.isArray(d)) {
        return d
          .map((x) => {
            if (x && typeof x === "object" && "msg" in x) return String((x as { msg: string }).msg);
            return JSON.stringify(x);
          })
          .join("; ");
      }
    }
  } catch {
    /* ignore */
  }
  return res.statusText || `HTTP ${res.status}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${apiBase()}${path.startsWith("/") ? path : `/${path}`}`;
  const method = (init?.method ?? "GET").toUpperCase();
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string>) };
  if (method !== "GET" && method !== "HEAD") {
    headers["Content-Type"] = headers["Content-Type"] ?? "application/json";
  }
  const res = await fetch(url, {
    ...init,
    headers,
  });
  if (!res.ok) {
    throw new ApiError(await parseErrorDetail(res), res.status);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health", { method: "GET" });
}

export async function postChat(body: { message: string; user_prompt?: string | null }): Promise<ChatResponse> {
  return requestJson<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      message: body.message,
      ...(body.user_prompt != null && body.user_prompt !== "" ? { user_prompt: body.user_prompt } : {}),
    }),
  });
}

export async function postReport(): Promise<ReportResponse> {
  return requestJson<ReportResponse>("/report", { method: "POST", body: "{}" });
}

export async function postReset(resumeRunDir?: string | null): Promise<ResetResponse> {
  return requestJson<ResetResponse>("/reset", {
    method: "POST",
    body: JSON.stringify(
      resumeRunDir && resumeRunDir.trim() ? { resume_run_dir: resumeRunDir.trim() } : {},
    ),
  });
}

export async function postPipeline(body?: {
  user_prompt?: string | null;
  follow_ups?: string[];
}): Promise<PipelineResponse> {
  const payload: Record<string, unknown> = {};
  if (body?.user_prompt != null && String(body.user_prompt).trim() !== "") {
    payload.user_prompt = String(body.user_prompt).trim();
  }
  if (body?.follow_ups?.length) {
    payload.follow_ups = body.follow_ups.filter((s) => String(s).trim());
  }
  return requestJson<PipelineResponse>("/pipeline", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
