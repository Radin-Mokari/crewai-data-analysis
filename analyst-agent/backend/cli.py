"""
CLI verification script for testing the orchestrator without the web frontend.
Usage: python -m backend.cli
"""

import asyncio
import sys
from pathlib import Path

from backend.orchestrator import AnalysisOrchestrator, inspect_dataset
from backend.tools import JupyterSessionTool
from backend.agents import create_specialists, create_manager
from backend.config import make_gemini_llm


async def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "Housing.csv"
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"[CLI] Using dataset: {dataset_path}")
    tool = JupyterSessionTool(output_dir="./results/charts")
    llm_medium = make_gemini_llm(640, 256)
    llm_long = make_gemini_llm(1200, 512)
    specialists = create_specialists(tool, llm_medium, llm_long)
    manager = create_manager(make_gemini_llm(640, 256))
    orchestrator = AnalysisOrchestrator(
        tool=tool,
        specialists=specialists,
        manager=manager,
        llm=llm_medium,
    )

    # Test Path A
    print("[CLI] Starting full analysis (Path A)...")
    result = await orchestrator.run(dataset_path, "Analyze this dataset completely")
    print(f"\n[CLI] Completed: {result['completed']}")
    print(f"[CLI] Cells created: {len(tool.get_cells())}")
    for cell in tool.get_cells():
        print(f"  [{cell['agent']}] Cell {cell['execution_count']}: {cell['code'][:60]}...")
    if result.get("charts"):
        print(f"[CLI] Charts: {result['charts']}")
    if result.get("errors"):
        print(f"[CLI] Errors: {result['errors']}")


if __name__ == "__main__":
    asyncio.run(main())
