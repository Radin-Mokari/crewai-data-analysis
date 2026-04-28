/**
 * Parse incremental SSE text into complete `data: {json}` payloads (split on blank line between events).
 * Handles both \n\n and \r\n\r\n delimiters for Windows/proxy compatibility.
 */
export function parseSseBuffer(buffer: string): { events: unknown[]; rest: string } {
  const events: unknown[] = [];
  // Split on any blank-line separator: \r\n\r\n or \n\n
  const parts = buffer.split(/\r?\n\r?\n/);
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    const dataLines = part.split(/\r?\n/).filter((l) => l.startsWith("data:"));
    if (dataLines.length === 0) continue;
    const payload = dataLines.map((l) => l.replace(/^data:\s?/, "")).join("\n");
    try {
      events.push(JSON.parse(payload));
    } catch {
      /* ignore malformed chunk */
    }
  }
  return { events, rest };
}
