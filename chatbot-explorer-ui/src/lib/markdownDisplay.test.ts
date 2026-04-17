import { describe, expect, it } from "vitest";
import {
  ensureBlankLineBeforeMarkdownTables,
  normalizeMarkdownTablesForGfm,
  repairSmashedMarkdownTableRows,
} from "./markdownDisplay";

describe("normalizeMarkdownTablesForGfm", () => {
  it("inserts blank line before a table after a bullet line", () => {
    const src = "- Point one\n| Col | Val |\n| --- | --- |";
    expect(ensureBlankLineBeforeMarkdownTables(src)).toBe("- Point one\n\n| Col | Val |\n| --- | --- |");
  });

  it("splits smashed header and separator row", () => {
    const smashed = "| a | b | |:--|--|";
    expect(repairSmashedMarkdownTableRows(smashed)).toBe("| a | b |\n|:--|--|");
  });

  it("does not mutate content inside code fences", () => {
    const src = "Text\n\n```\n| a | b | |:--|\n```\n\n| x | y |\n| - | - |";
    const out = normalizeMarkdownTablesForGfm(src);
    expect(out).toContain("```\n| a | b | |:--|");
    expect(out).toMatch(/\n\n\| x \| y \|/);
  });
});
