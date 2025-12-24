#!/usr/bin/env python3
"""
Minimal runner for CrewAI Data Analysis
Run this file to execute the analysis workflow
"""


import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure Gemini as the default LLM for CrewAI
# This must be set BEFORE importing crewai_data_analysis
os.environ["CREWAI_LLM_MODEL"] = "gemini-2.5-flash"


from crewai_data_analysis import DataAnalysisWorkflow, create_sample_dataset


def main():
    """Main entry point."""
    
    print(f"\n{'='*70}")
    print("CREWAI DATA ANALYSIS WORKFLOW")
    print(f"{'='*70}\n")
    
    # Get configuration from .env
    dataset_path = os.getenv("DATASET_PATH", "sample_data.csv")
    output_dir = os.getenv("OUTPUT_DIR", "./analysis_results")
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Verify API key is set
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not set!")
        print("\nFix: Open .env file and add your API key:")
        print("  GEMINI_API_KEY=your_actual_key_here")
        return
    
    print(f"✓ API Key configured")
    print(f"✓ Using LLM: gemini-2.5-flash")
    
    # Create sample dataset if it doesn't exist
    if not Path(dataset_path).exists():
        print(f"\nCreating sample dataset: {dataset_path}")
        create_sample_dataset(dataset_path)
        print(f"✓ Sample dataset created")
    else:
        print(f"✓ Using existing dataset: {dataset_path}")
    
    # Initialize workflow
    print(f"\nInitializing workflow...")
    print(f"  Input: {dataset_path}")
    print(f"  Output: {output_dir}\n")
    
    workflow = DataAnalysisWorkflow(
        dataset_path=dataset_path,
        output_dir=output_dir
    )
    
    # Run analysis
    print("Starting analysis pipeline...\n")
    try:
        results = workflow.run_sequential_pipeline()
        
        # Generate report
        print("\nGenerating HTML report...")
        report_path = workflow.generate_html_report()
        
        print(f"\n{'='*70}")
        print("✅ ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📊 Report saved to:")
        print(f"  {Path(report_path).absolute()}\n")
        print(f"📁 Output directory:")
        print(f"  {Path(output_dir).absolute()}\n")
        
        return report_path
    
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ ANALYSIS FAILED")
        print(f"{'='*70}")
        print(f"\nError: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check GEMINI_API_KEY in .env file")
        print(f"  2. Verify dataset file exists: {dataset_path}")
        print(f"  3. Check internet connection")
        print(f"  4. Review error message above")
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
