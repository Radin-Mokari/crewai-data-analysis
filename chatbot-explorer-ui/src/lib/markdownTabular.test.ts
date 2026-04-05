import { describe, expect, it } from "vitest";
import { looksLikeTabularPlaintextLine, wrapTabularPlaintextInFences } from "./markdownTabular";

describe("markdownTabular", () => {
  it("detects pandas describe-style lines", () => {
    expect(looksLikeTabularPlaintextLine("count  20640.000000  20640.000000        20640.000000")).toBe(true);
    expect(looksLikeTabularPlaintextLine("mean    -119.569704     35.631861")).toBe(true);
  });

  it("rejects markdown pipes and headings", () => {
    expect(looksLikeTabularPlaintextLine("| a | b |")).toBe(false);
    expect(looksLikeTabularPlaintextLine("## Section")).toBe(false);
  });

  it("wraps contiguous tabular lines in text fence", () => {
    const raw = `Some intro.

count  20640.0  20640.0
mean     -119.5     35.6

After.`;
    const out = wrapTabularPlaintextInFences(raw);
    expect(out).toContain("```text");
    expect(out).toContain("count  20640.0");
    expect(out).toContain("After.");
  });

  it("does not touch existing fenced blocks", () => {
    const raw = "```python\nx = 1\n```\n\nmean   1.0   2.0\nstd   0.1   0.2";
    const out = wrapTabularPlaintextInFences(raw);
    expect(out).toContain("```python");
    expect(out).toContain("```text");
  });
});
