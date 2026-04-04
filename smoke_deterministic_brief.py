#!/usr/bin/env python3
"""No-LLM smoke test: load CSV, init session, write dataset_brief.txt ."""

import sys
from pathlib import Path

from crewai_data_analysis import PythonSessionTool, compute_dataset_brief


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Housing.csv"
    p = Path(csv_path)
    if not p.exists():
        print(f"File not found: {p}")
        return 1
    out = Path("./analysis_results") / "smoke_brief_only"
    out.mkdir(parents=True, exist_ok=True)
    ex = PythonSessionTool(output_dir=str(out / "charts"))
    ex.init_session(str(p.resolve()))
    text, brief = compute_dataset_brief(ex, out)
    assert (out / "dataset_brief.txt").exists()
    print(text[:2500])
    print("\n--- dict keys:", list(brief.keys()))
    print("is_time_series:", brief.get("is_time_series"))
    print("[OK] dataset_brief.txt written to", out / "dataset_brief.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
