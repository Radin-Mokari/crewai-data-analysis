/**
 * Parse incremental SSE text into complete `data: {json}` payloads (split on blank line between events).
 */
export function parseSseBuffer(buffer: string): { events: unknown[]; rest: string } {
  const events: unknown[] = [];
  const parts = buffer.split(/\n\n/);
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    const dataLines = part.split("\n").filter((l) => l.startsWith("data:"));
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
