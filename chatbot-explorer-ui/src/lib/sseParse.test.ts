import { describe, expect, it } from "vitest";
import { parseSseBuffer } from "./sseParse";

describe("parseSseBuffer", () => {
  it("parses one SSE event", () => {
    const raw = 'data: {"type":"guardrail","message":"stop"}\n\n';
    const { events, rest } = parseSseBuffer(raw);
    expect(rest).toBe("");
    expect(events).toHaveLength(1);
    expect(events[0]).toEqual({ type: "guardrail", message: "stop" });
  });

  it("preserves incomplete trailing buffer", () => {
    const raw = 'data: {"a":1}';
    const { events, rest } = parseSseBuffer(raw);
    expect(events).toHaveLength(0);
    expect(rest).toBe(raw);
  });

  it("parses multiple events and leaves rest", () => {
    const raw =
      'data: {"type":"x"}\n\n' + 'data: {"type":"final","run_id":"r"}\n\n' + "partial";
    const { events, rest } = parseSseBuffer(raw);
    expect(events).toHaveLength(2);
    expect((events[1] as { type: string }).type).toBe("final");
    expect(rest).toBe("partial");
  });
});
