/**
 * Normalizes agent / report markdown so ReactMarkdown + remark-gfm render headings, tables, and images.
 */

/** Apply `fn` only to segments outside ``` fenced blocks. */
function transformOutsideCodeFences(markdown: string, fn: (chunk: string) => string): string {
  const chunks: string[] = [];
  let pos = 0;
  while (pos < markdown.length) {
    const start = markdown.indexOf("```", pos);
    if (start === -1) {
      chunks.push(fn(markdown.slice(pos)));
      break;
    }
    chunks.push(fn(markdown.slice(pos, start)));
    const afterOpen = start + 3;
    const close = markdown.indexOf("```", afterOpen);
    if (close === -1) {
      chunks.push(markdown.slice(start));
      break;
    }
    chunks.push(markdown.slice(start, close + 3));
    pos = close + 3;
  }
  return chunks.join("");
}

const TABLE_ROW_LIKE = /^\|.*\|\s*$/;
const TABLE_SEP_LIKE = /^\|?\s*:?[-=:| ]+\|\s*$/;

/**
 * GFM needs a blank line before a table; list items swallow `|` rows otherwise.
 */
export function ensureBlankLineBeforeMarkdownTables(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  const isTableLine = (line: string) => {
    const t = line.trimEnd();
    return TABLE_ROW_LIKE.test(t) || TABLE_SEP_LIKE.test(t);
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const prev = out.length ? out[out.length - 1]! : "";
    if (isTableLine(line) && prev.trim() !== "" && !isTableLine(prev)) {
      if (out.length && out[out.length - 1]!.trim() !== "") {
        out.push("");
      }
    }
    out.push(line);
  }
  return out.join("\n");
}

/**
 * Models sometimes concatenate the separator row onto the header row, e.g. `| a | b | |:---|`.
 * Split before `| |:` or `| | -` style separator starts.
 */
export function repairSmashedMarkdownTableRows(text: string): string {
  const lines = text.split("\n");
  return lines
    .map((line) => {
      const pipes = line.match(/\|/g)?.length ?? 0;
      if (pipes < 4) return line;
      // e.g. `| a | b | |:--|--|` → newline before `|:--` separator row
      return line.replace(/\|\s+\|(?=:)/g, "|\n|");
    })
    .join("\n");
}

/** Table-oriented GFM fixes (skips fenced code). */
export function normalizeMarkdownTablesForGfm(markdown: string): string {
  return transformOutsideCodeFences(markdown, (chunk) =>
    ensureBlankLineBeforeMarkdownTables(repairSmashedMarkdownTableRows(chunk)),
  );
}

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
  out = normalizeMarkdownTablesForGfm(out);
  out = rewriteEmbeddedChartPaths(out, runId);
  return out;
}
