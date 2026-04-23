/**
 * FastAPI backend (server.py). Default: same-origin `/api/*` via Vite dev proxy → http://127.0.0.1:8765.
 * Override with VITE_API_BASE_URL (no trailing slash), e.g. http://127.0.0.1:8765
 */

import { parseSseBuffer } from "@/lib/sseParse";

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
  /** Human-readable markdown summary generated after specialists run. Primary content for the bot reply. */
  manager_reply?: string;
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

/** SSE payloads from supervisor (before `final`). Chain-of-thoughts events. */
export type SupervisorStreamEvent =
  | { type: "manager_decision"; instruction?: string; rationale?: string; next_agent: string }
  | { type: "specialist_start"; step: number; agent: string }
  | { type: "specialist_complete"; step: number; agent: string; excerpt?: string; agent_code?: string }
  | { type: "manager_message"; text: string }
  | { type: "manager_summary"; text: string }
  | { type: "guardrail"; message: string }
  | { type: "heartbeat" };

/**
 * Infrastructure log events — a *separate* channel from `SupervisorStreamEvent`.
 * These feed the Logs sidebar only and are NOT shown in chain-of-thoughts.
 * No inputs/outputs/reasoning content — just "what happened" (agent started,
 * task retry, guardrail, specialist failed, etc.).
 */
export type LogLevel = "info" | "warn" | "error";
export type LogCategory =
  | "workflow"
  | "manager"
  | "specialist"
  | "task"
  | "tool"
  | "guardrail";

export type LogEvent = {
  type: "log";
  ts: string;
  level: LogLevel;
  category: LogCategory;
  event: string;
  message: string;
  agent?: string;
  step?: number;
  task?: string;
  attempt?: number;
  total?: number;
  turn?: number;
  outcome?: string;
  next_agent?: string;
};

export type PipelineStreamFinal = {
  ok: boolean;
  run_id: string;
  specialist_steps: number;
  lines: string[];
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

async function readSsePost(
  path: string,
  jsonBody: Record<string, unknown>,
  onEvent: (obj: Record<string, unknown>) => void,
  onLog?: (log: LogEvent) => void,
): Promise<Record<string, unknown>> {
  const url = `${apiBase()}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(jsonBody),
  });
  if (!res.ok) {
    throw new ApiError(await parseErrorDetail(res), res.status);
  }
  const reader = res.body?.getReader();
  if (!reader) {
    throw new ApiError("No response body", 500);
  }
  const decoder = new TextDecoder();
  let buf = "";
  let finalPayload: Record<string, unknown> | null = null;
  const dispatch = (obj: Record<string, unknown>) => {
    const t = obj.type;
    if (t === "final") {
      finalPayload = obj;
    } else if (t === "error") {
      throw new ApiError(String(obj.message ?? "Stream error"), 500);
    } else if (t === "log") {
      if (onLog) onLog(obj as unknown as LogEvent);
      // Intentionally NOT forwarded to onEvent — chain-of-thoughts channel is
      // kept unchanged. Log events feed the Logs sidebar only.
    } else {
      onEvent(obj);
    }
  };
  while (true) {
    const { done, value } = await reader.read();
    buf += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const { events, rest } = parseSseBuffer(buf);
    buf = rest;
    for (const raw of events) {
      if (!raw || typeof raw !== "object") continue;
      dispatch(raw as Record<string, unknown>);
    }
    if (done) break;
  }
  const { events: tailEvents } = parseSseBuffer(buf + "\n\n");
  for (const raw of tailEvents) {
    if (!raw || typeof raw !== "object") continue;
    dispatch(raw as Record<string, unknown>);
  }
  if (!finalPayload) {
    throw new ApiError("Stream ended without final event", 500);
  }
  return finalPayload;
}

/** POST /chat/stream — SSE; `onEvent` receives supervisor events; resolves with same shape as POST /chat (plus `type` on final). */
export async function postChatStream(
  body: { message: string; user_prompt?: string | null },
  onEvent: (evt: SupervisorStreamEvent) => void,
  onLog?: (log: LogEvent) => void,
): Promise<ChatResponse> {
  const payload: Record<string, unknown> = {
    message: body.message,
  };
  if (body.user_prompt != null && body.user_prompt !== "") {
    payload.user_prompt = body.user_prompt;
  }
  const fin = await readSsePost(
    "/chat/stream",
    payload,
    (obj) => {
      onEvent(obj as unknown as SupervisorStreamEvent);
    },
    onLog,
  );
  const { type: _t, ...rest } = fin;
  return rest as unknown as ChatResponse;
}

/** POST /pipeline/stream — SSE for full batch; resolves when pipeline completes. */
export async function postPipelineStream(
  body: { user_prompt?: string | null; follow_ups?: string[] },
  onEvent: (evt: SupervisorStreamEvent) => void,
  onLog?: (log: LogEvent) => void,
): Promise<PipelineStreamFinal> {
  const payload: Record<string, unknown> = {};
  if (body.user_prompt != null && String(body.user_prompt).trim() !== "") {
    payload.user_prompt = String(body.user_prompt).trim();
  }
  if (body.follow_ups?.length) {
    payload.follow_ups = body.follow_ups.filter((s) => String(s).trim());
  }
  const fin = await readSsePost(
    "/pipeline/stream",
    payload,
    (obj) => {
      onEvent(obj as unknown as SupervisorStreamEvent);
    },
    onLog,
  );
  const { type: _t, ...rest } = fin;
  return rest as unknown as PipelineStreamFinal;
}

export interface RunCodeResponse {
  success: boolean;
  stdout: string;
  error: string | null;
  charts: string[];
  state_flags?: Record<string, boolean>;
}

export async function postRunCode(code: string): Promise<RunCodeResponse> {
  const url = resolveArtifactUrl("/run_code");
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
  if (!response.ok) {
    const txt = await response.text();
    throw new ApiError(txt, response.status);
  }
  return response.json();
}
