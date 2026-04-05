/**
 * Pandas `to_string()` / stdout tables are plain text with spaces — ReactMarkdown renders
 * them as <p> and destroys columns. Wrap those runs in ```text fences so they render as <pre>.
 */

/** Line looks like fixed-width tabular / numeric grid (describe, corr, head rows). */
export function looksLikeTabularPlaintextLine(line: string): boolean {
  const t = line.trimEnd();
  if (t.length === 0) return false;
  if (t.startsWith("#")) return false;
  if (t.startsWith("|")) return false;
  if (/^[-*_]{3,}\s*$/.test(t)) return false;

  if (/^(count|mean|std|min|max|25%|50%|75%|dtype:|Name:)\b/i.test(t.trim())) return true;

  if (t.length < 16) return false;

  // Multiple internal spaces + digits (column-aligned numbers)
  if (/\s{2,}/.test(line) && /[\d.]/.test(t)) {
    return true;
  }

  // Correlation / matrix rows: leading spaces then tokens
  if (/^\s{2,}\S/.test(line) && /[\d.-]{4,}/.test(t)) {
    return true;
  }

  // Header row of wide tables: many spaced columns (feature names), few/no digits
  if (
    t.length >= 40 &&
    /\s{2,}/.test(line) &&
    (t.split(/\s{2,}/).filter(Boolean).length >= 4 || t.split(/\s+/).length >= 8)
  ) {
    return true;
  }

  return false;
}

function isStrongSingleLineTabular(line: string): boolean {
  return line.length >= 100 && /\s{3,}/.test(line) && /[\d.]/.test(line);
}

function wrapRegion(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  let i = 0;

  while (i < lines.length) {
    if (!looksLikeTabularPlaintextLine(lines[i])) {
      out.push(lines[i]);
      i++;
      continue;
    }

    const block: string[] = [];
    while (i < lines.length) {
      const line = lines[i];
      if (looksLikeTabularPlaintextLine(line)) {
        block.push(line);
        i++;
        continue;
      }
      if (line.trim() === "" && block.length > 0) {
        const next = i + 1 < lines.length ? lines[i + 1] : "";
        if (next && looksLikeTabularPlaintextLine(next)) {
          block.push(line);
          i++;
          continue;
        }
      }
      break;
    }

    const shouldFence =
      block.length >= 2 || (block.length === 1 && isStrongSingleLineTabular(block[0]!));

    if (shouldFence) {
      out.push("```text");
      out.push(...block);
      out.push("```");
    } else {
      out.push(...block);
    }
  }

  return out.join("\n");
}

/**
 * Wrap likely pandas/stdout table blocks in ```text fences. Skips existing fenced code.
 */
export function wrapTabularPlaintextInFences(markdown: string): string {
  const chunks: string[] = [];
  let pos = 0;
  while (pos < markdown.length) {
    const start = markdown.indexOf("```", pos);
    if (start === -1) {
      chunks.push(wrapRegion(markdown.slice(pos)));
      break;
    }
    chunks.push(wrapRegion(markdown.slice(pos, start)));
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
