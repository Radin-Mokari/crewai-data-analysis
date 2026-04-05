/**
 * Normalizes agent / report markdown so ReactMarkdown + remark-gfm render headings, tables, and images.
 */

/** Extract `20260405_183642` from a run folder path ending in `run_20260405_183642`. */
export function runIdFromRunDir(runDir: string): string | null {
  const parts = runDir.replace(/\\/g, "/").split("/").filter(Boolean);
  const last = parts[parts.length - 1];
  if (last?.startsWith("run_")) return last.slice(4);
  return null;
}

/**
 * If every non-empty line shares a large common indent (>= 4 spaces), dedent — otherwise
 * the block is parsed as an indented code block and headings/tables won't render.
 */
export function dedentMarkdownIfNeeded(text: string): string {
  const lines = text.split("\n");
  const nonEmpty = lines.filter((l) => l.trim().length > 0);
  if (nonEmpty.length === 0) return text;
  let min = Infinity;
  for (const line of nonEmpty) {
    const m = /^(\s*)/.exec(line);
    const w = m ? m[1].length : 0;
    min = Math.min(min, w);
  }
  if (min === Infinity || min < 4) return text;
  return lines.map((l) => (l.trim().length ? l.slice(min) : l)).join("\n");
}

/**
 * Replace markdown image targets that point at local `.../charts/chart_*.png` paths with
 * `/artifacts/{runId}/charts/{file}` so the UI can proxy-load them.
 */
export function rewriteEmbeddedChartPaths(markdown: string, runId: string): string {
  if (!runId.trim()) return markdown;
  return markdown.replace(/!\[([^\]]*)\]\([^)]*\)/g, (full, alt: string) => {
    const m = /chart_\d+_\d+\.png/i.exec(full);
    if (!m) return full;
    const filename = m[0];
    return `![${alt}](/artifacts/${runId}/charts/${filename})`;
  });
}

/** Apply both transforms for chat or report bodies. */
export function prepareMarkdownForChat(markdown: string, runId: string): string {
  let out = markdown.trimEnd();
  out = dedentMarkdownIfNeeded(out);
  out = rewriteEmbeddedChartPaths(out, runId);
  return out;
}
