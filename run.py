#!/usr/bin/env python3
"""
Minimal runner for CrewAI Data Analysis
This file triggers the analysis workflow
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.setrecursionlimit(5000)

load_dotenv()

os.environ["CREWAI_LLM_MODEL"] = "gemini-2.5-flash"

from crewai_data_analysis import DataAnalysisWorkflow

agentops_key = os.getenv("AGENTOPS_API_KEY")
if agentops_key:
    import agentops
    agentops.init(
        api_key=agentops_key,
        default_tags=['crewai']
    )
    print("OK, AgentOps monitoring enabled - view at https://app.agentops.ai")


def main():
    """Main entry point."""

    cli = argparse.ArgumentParser(description="CrewAI data analysis (sequential or dynamic supervisor).")
    cli.add_argument(
        "--workflow-mode",
        choices=["dynamic", "sequential"],
        default=None,
        help="Override WORKFLOW_MODE from .env",
    )
    cli.add_argument(
        "--resume",
        default=None,
        help="Path to an existing run_* folder (overrides RESUME_RUN_DIR)",
    )
    cli.add_argument(
        "--follow-up",
        action="append",
        dest="follow_ups",
        default=[],
        help="Extra user message merged into supervisor chat (dynamic mode); repeatable",
    )
    cli.add_argument(
        "--interactive",
        action="store_true",
        help="After dynamic pipeline, stay in stdin loop (CHAT/specialist/DONE); /report, exit",
    )
    cli.add_argument(
        "--no-interactive",
        action="store_true",
        help="After dynamic run, exit immediately (skip stdin manager chat and on-demand /report).",
    )
    args, _unknown = cli.parse_known_args()

    print(f"\n{'='*70}")
    print("CREWAI DATA ANALYSIS WORKFLOW")
    print(f"{'='*70}\n")

    dataset_path = os.getenv("DATASET_PATH")
    output_dir = os.getenv("OUTPUT_DIR", "./analysis_results")
    api_key = os.getenv("GEMINI_API_KEY")
    workflow_mode = (
        args.workflow_mode or os.getenv("WORKFLOW_MODE", "dynamic")
    ).strip().lower()
    env_interactive = os.getenv("INTERACTIVE_SESSION", "").strip().lower()
    if args.no_interactive:
        interactive = False
    elif args.interactive:
        interactive = True
    elif env_interactive in ("0", "false", "no", "off"):
        interactive = False
    elif env_interactive in ("1", "true", "yes"):
        interactive = True
    else:
        # Default: keep process alive for manager chat after dynamic run (plan: same kernel + JSONL).
        interactive = workflow_mode == "dynamic"
    user_prompt = os.getenv(
        "USER_ANALYSIS_PROMPT",
        "Perform thorough exploratory analysis and preprocessing; highlight data quality issues and modeling recommendations.",
    ).strip()
    resume_run_dir = (args.resume or os.getenv("RESUME_RUN_DIR", "") or "").strip()

    if not dataset_path:
        print("[ERROR] DATASET_PATH not set!")
        print("\nFix: Open .env file and add your dataset path:")
        print("  DATASET_PATH=your_dataset.csv")
        return

    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set!")
        print("\nFix: Open .env file and add your API key:")
        print("  GEMINI_API_KEY=your_actual_key_here")
        return

    print("[OK] Gemini API Key configured")
    print("[OK] Using LLM: gemini-2.5-flash")

    if not Path(dataset_path).exists():
        print(f"\n[ERROR] Dataset not found on disk: {dataset_path}")
        print("Fix one of these:")
        print("  - Place your CSV in this project folder with that exact name, or")
        print("  - Update DATASET_PATH in .env to the correct relative path.")
        return
    else:
        print(f"[OK] Using dataset: {dataset_path}")

    print("\nInitializing workflow...")
    print(f"  Input: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {workflow_mode} (set WORKFLOW_MODE=sequential for legacy pipeline)")
    if resume_run_dir:
        print(f"  Resume: {resume_run_dir}")
    if interactive and workflow_mode == "dynamic":
        print("  Interactive: enabled (stdin after pipeline; /report, exit)")
    print()

    workflow = DataAnalysisWorkflow(
        dataset_path=dataset_path,
        output_dir=output_dir,
        resume_from=resume_run_dir or None,
    )

    print("Starting analysis pipeline...\n")
    try:
        if workflow_mode == "sequential":
            workflow.run_sequential_pipeline()
        else:
            fu = [x for x in (args.follow_ups or []) if str(x).strip()]
            workflow.run_dynamic_team_pipeline(
                user_prompt=user_prompt,
                followup_messages=fu or None,
                skip_terminal_reporter=bool(interactive),
            )
            if interactive:
                workflow.run_interactive_session(user_prompt=user_prompt)

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