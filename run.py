#!/usr/bin/env python3
"""
Minimal runner for CrewAI Data Analysis
Run this file to execute the analysis workflow
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix recursion error from rich library
sys.setrecursionlimit(5000)

# Load environment variables from .env file
load_dotenv()

# Configure Gemini as the default LLM for CrewAI
# This must be set BEFORE importing crewai_data_analysis
os.environ["CREWAI_LLM_MODEL"] = "gemini-2.5-flash"

from crewai_data_analysis import DataAnalysisWorkflow

# Initialize AgentOps for workflow observability (if API key is set)
# NOTE: Must be AFTER CrewAI imports to avoid circular import error
agentops_key = os.getenv("AGENTOPS_API_KEY")
if agentops_key:
    import agentops
    agentops.init(
        api_key=agentops_key,
        default_tags=['crewai']
    )
    print("[OK] AgentOps monitoring enabled - view at https://app.agentops.ai")


def main():
    """Main entry point."""

    print(f"\n{'='*70}")
    print("CREWAI DATA ANALYSIS WORKFLOW")
    print(f"{'='*70}\n")

    # Get configuration from .env
    # Set DATASET_PATH in .env, e.g. DATASET_PATH=CarSales_Dataset.csv
    dataset_path = os.getenv("DATASET_PATH", "CarSales_Dataset.csv")
    output_dir = os.getenv("OUTPUT_DIR", "./analysis_results")
    api_key = os.getenv("GEMINI_API_KEY")

    # Verify API key is set
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set!")
        print("\nFix: Open .env file and add your API key:")
        print("  GEMINI_API_KEY=your_actual_key_here")
        return

    print("[OK] Gemini API Key configured")
    print("[OK] Using LLM: gemini-2.5-flash")

    # Require that the dataset already exists; do NOT create any sample CSV
    if not Path(dataset_path).exists():
        print(f"\n[ERROR] Dataset not found on disk: {dataset_path}")
        print("Fix one of these:")
        print("  - Place your CSV in this project folder with that exact name, or")
        print("  - Update DATASET_PATH in .env to the correct relative path.")
        return
    else:
        print(f"[OK] Using dataset: {dataset_path}")

    # Initialize workflow
    print("\nInitializing workflow...")
    print(f"  Input: {dataset_path}")
    print(f"  Output: {output_dir}\n")

    workflow = DataAnalysisWorkflow(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    # Run analysis
    print("Starting analysis pipeline...\n")
    try:
        results = workflow.run_sequential_pipeline()

        # Generate markdown report
        print("\nGenerating markdown report...")
        report_path = workflow.generate_markdown_report()

        print(f"\n{'='*70}")
        print("[SUCCESS] ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"\n[RUN ID] {workflow.run_id}")
        print("\n[REPORT] Report saved to:")
        print(f"  {Path(report_path).absolute()}\n")
        print("[OUTPUT] Run output directory:")
        print(f"  {workflow.run_output_dir.absolute()}\n")

        return report_path

    except Exception as e:
        print(f"\n{'='*70}")
        print("[FAILED] ANALYSIS FAILED")
        print(f"{'='*70}")
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print(f"  1. Check GEMINI_API_KEY in .env file")
        print(f"  2. Verify dataset file exists: {dataset_path}")
        print("  3. Check internet connection")
        print("  4. Review error message above")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()