import { describe, expect, it } from "vitest";
import {
  extractJsonArray,
  formatChatReply,
  parseVisualizationChartMeta,
  proseOutsideJsonArray,
} from "./formatChatReply";

describe("formatChatReply helpers", () => {
  it("extractJsonArray parses embedded array", () => {
    const text = `Here is data:\n[{"chart_path": "C:/x/chart_1.png", "title": "T", "description": "D"}]\n`;
    const arr = extractJsonArray(text);
    expect(arr).toHaveLength(1);
    expect((arr![0] as { title: string }).title).toBe("T");
  });

  it("parseVisualizationChartMeta reads chart fields", () => {
    const ex = `[{"chart_path": "C:\\\\a\\\\chart_177_1.png", "title": "Corr", "description": "Heatmap"}]`;
    const m = parseVisualizationChartMeta(ex);
    expect(m?.[0]?.filename).toBe("chart_177_1.png");
    expect(m?.[0]?.title).toBe("Corr");
  });

  it("proseOutsideJsonArray strips array", () => {
    expect(proseOutsideJsonArray("Intro\n\n[{}]")).toBe("Intro");
    expect(proseOutsideJsonArray("[{}]\ntrailer")).toBe("trailer");
  });

  it("formatChatReply pairs charts with metadata", () => {
    const md = formatChatReply(
      {
        outcome: "await_user",
        lines: ["[MANAGER]\nok"],
        run_id: "20260101_120000",
        specialist_steps: [
          { agent: "cleaning", excerpt: "done" },
          {
            agent: "visualization",
            excerpt: `[{"chart_path": "/r/c.png", "title": "Heat", "description": "x"}]`,
          },
        ],
        chart_urls: ["/artifacts/20260101_120000/charts/c.png"],
      },
      (p) => `/api${p}`,
    );
    expect(md).toContain("### cleaning");
    expect(md).toContain("done");
    expect(md).not.toContain("```");
    expect(md).toContain("#### Heat");
    expect(md).toContain("![Heat]");
    expect(md).toContain("/api/artifacts/20260101_120000/charts/c.png");
  });

  it("formatChatReply keeps specialist + viz captions when manager_reply is set", () => {
    const md = formatChatReply(
      {
        outcome: "await_user",
        lines: [],
        run_id: "r1",
        manager_reply: "Here is the analysis summary.",
        specialist_steps: [
          { agent: "eda", excerpt: "stats" },
          {
            agent: "visualization",
            excerpt: `[{"chart_path": "/r/c.png", "title": "Heat", "description": "x"}]`,
          },
        ],
        chart_urls: ["/artifacts/r1/charts/c.png"],
      },
      (p) => `/api${p}`,
    );
    expect(md).toContain("analysis summary");
    expect(md).toContain("### eda");
    expect(md).toContain("#### Heat");
  });

  it("formatChatReply drops duplicate [MANAGER] log when manager_reply matches", () => {
    const report = "# Report\n\nBody here.";
    const md = formatChatReply(
      {
        outcome: "await_user",
        lines: [`[MANAGER]\n${report}`],
        run_id: "r1",
        manager_reply: report,
        specialist_steps: [{ agent: "eda", excerpt: "eda only" }],
        chart_urls: [],
      },
      (p) => p,
    );
    expect(md.match(/# Report/g)?.length).toBe(1);
    expect(md).toContain("### eda");
  });

  it("formatChatReply omits reporter excerpt when manager_reply is long markdown synthesis", () => {
    const mgr = "# Executive Summary\n\n" + "x".repeat(1300);
    const md = formatChatReply(
      {
        outcome: "await_user",
        lines: [],
        run_id: "r1",
        manager_reply: mgr,
        specialist_steps: [
          { agent: "eda", excerpt: "eda out" },
          { agent: "reporter", excerpt: "# Executive Summary\n\n(dup)" },
        ],
        chart_urls: [],
      },
      (p) => p,
    );
    expect(md).toContain("### eda");
    expect(md).not.toContain("### reporter");
  });

  it("formatChatReply omits specialist excerpts when omitSpecialistExcerpts", () => {
    const md = formatChatReply(
      {
        outcome: "await_user",
        lines: ["[MANAGER]\nhi"],
        run_id: "r1",
        specialist_steps: [{ agent: "eda", excerpt: "long excerpt" }],
        chart_urls: [],
      },
      (p) => p,
      { omitSpecialistExcerpts: true },
    );
    expect(md).not.toContain("This turn: specialist");
    expect(md).not.toContain("### eda");
    expect(md).toContain("[MANAGER]");
  });
});
