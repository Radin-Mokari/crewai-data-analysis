import type { ChatResponse } from "@/lib/api";
import { prepareMarkdownForChat } from "@/lib/markdownDisplay";

/** Windows or POSIX path basename */
function pathBasename(p: string): string {
  const s = p.replace(/\\/g, "/");
  const parts = s.split("/");
  return parts[parts.length - 1] || s;
}

export type ChartMeta = {
  title?: string;
  description?: string;
  /** Basename from agent JSON e.g. chart_123_1.png */
  filename?: string;
};

/**
 * Extract first top-level JSON array from text (e.g. visualization agent chart metadata).
 */
export function extractJsonArray(text: string): unknown[] | null {
  const t = text;
  const start = t.indexOf("[");
  if (start === -1) return null;
  let depth = 0;
  let end = -1;
  for (let i = start; i < t.length; i++) {
    const c = t[i];
    if (c === "[") depth++;
    else if (c === "]") {
      depth--;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  if (end === -1) return null;
  try {
    const parsed = JSON.parse(t.slice(start, end + 1)) as unknown;
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function parseVisualizationChartMeta(excerpt: string): ChartMeta[] | null {
  const arr = extractJsonArray(excerpt);
  if (!arr?.length) return null;
  const out: ChartMeta[] = [];
  for (const item of arr) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const chartPath = typeof o.chart_path === "string" ? o.chart_path : "";
    out.push({
      title: typeof o.title === "string" ? o.title : undefined,
      description: typeof o.description === "string" ? o.description : undefined,
      filename: chartPath ? pathBasename(chartPath) : undefined,
    });
  }
  return out.length ? out : null;
}

/** Text outside the first top-level `[...]` JSON array (prose before/after chart metadata). */
export function proseOutsideJsonArray(excerpt: string): string {
  const t = excerpt.trim();
  const start = t.indexOf("[");
  if (start === -1) return t;
  let depth = 0;
  let end = -1;
  for (let i = start; i < t.length; i++) {
    const c = t[i];
    if (c === "[") depth++;
    else if (c === "]") {
      depth--;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  if (end === -1) return t;
  const before = t.slice(0, start).trim();
  const after = t.slice(end + 1).trim();
  return [before, after].filter(Boolean).join("\n\n").trim();
}

function buildChartsSection(
  urls: string[],
  vizExcerpt: string,
  resolveArtifactUrl: (path: string) => string,
): string {
  const lines: string[] = ["## Charts"];
  const meta = parseVisualizationChartMeta(vizExcerpt);

  if (meta && meta.length > 0 && urls.length > 0) {
    const intro = proseOutsideJsonArray(vizExcerpt);
    if (intro) {
      lines.push(intro);
      lines.push("");
    }
    for (let i = 0; i < urls.length; i++) {
      const u = urls[i];
      const base = pathBasename(u);
      let m = meta.find((e) => e.filename && e.filename === base);
      if (!m) m = meta[i];
      const title = (m?.title || `Chart ${i + 1}`).trim();
      const desc = (m?.description || "").trim();
      const src = resolveArtifactUrl(u);
      lines.push(`#### ${title}`);
      if (desc) {
        lines.push(desc);
      }
      const alt = title.replace(/[\]]/g, "");
      lines.push(`![${alt}](${src})`);
      lines.push("");
    }
    return lines.join("\n\n").trimEnd();
  }

  if (vizExcerpt) {
    lines.push(vizExcerpt);
    lines.push("");
    for (const u of urls) {
      lines.push(`![chart](${resolveArtifactUrl(u)})`);
      lines.push("");
    }
    return lines.join("\n\n").trimEnd();
  }

  for (const u of urls) {
    lines.push(`![chart](${resolveArtifactUrl(u)})`);
    lines.push("");
  }
  return lines.join("\n\n").trimEnd();
}

/**
 * Formats POST /chat JSON into a single markdown string for ReactMarkdown.
 * Does not wrap specialist excerpts in code fences (so **bold** and # headings render).
 * Visualization step excerpt is paired with chart URLs when JSON metadata is present.
 */
export function formatChatReply(
  data: ChatResponse,
  resolveArtifactUrl: (path: string) => string,
): string {
  const linesBlock = data.lines.map((l) => l.trimEnd()).filter(Boolean).join("\n\n");
  const meta = `\n\n---\n_Outcome:_ \`${data.outcome}\` · _run_id:_ \`${data.run_id}\``;

  const sections: string[] = [];
  if (linesBlock) {
    sections.push(linesBlock);
  }

  const steps = data.specialist_steps ?? [];
  const urls = data.chart_urls ?? [];

  const vizStep = [...steps].reverse().find((s) => s.agent === "visualization");
  const vizExcerpt = vizStep?.excerpt?.trim() ?? "";

  const otherSteps = steps.filter((s) => s.agent !== "visualization");

  if (otherSteps.length > 0) {
    const parts = otherSteps.map((s) => {
      const agent = s.agent || "specialist";
      const ex = (s.excerpt || "").trim();
      if (!ex) {
        return `### ${agent}\n\n_(no excerpt)_`;
      }
      return `### ${agent}\n\n${ex}`;
    });
    sections.push("## This turn: specialist output\n\n" + parts.join("\n\n"));
  }

  if (urls.length > 0) {
    sections.push(buildChartsSection(urls, vizExcerpt, resolveArtifactUrl));
  }

  if (sections.length === 0) {
    return prepareMarkdownForChat(`_(No log output from this turn.)_${meta}`, data.run_id);
  }
  const raw = `${sections.join("\n\n")}${meta}`;
  return prepareMarkdownForChat(raw, data.run_id);
}
