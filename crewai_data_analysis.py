# ============================================================================
# COMPLETE CREWAI DATA ANALYSIS WORKFLOW
# Multi-Agent Sequential Pipeline for Gemini (Rate-Limited + Token-Optimized)
# ============================================================================

import os
import subprocess
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import Field

# LLM configuration for Gemini
import google.generativeai as genai


# ============================================================================
# PART 1: CUSTOM TOOLS
# ============================================================================

class PythonCodeExecutorTool(BaseTool):
    """
    Safely executes Python code in isolated subprocess with timeout.
    
    Returns structured output:
    - code: The executed code
    - stdout: Output from code
    - stderr: Error messages
    - success: Whether execution succeeded
    - execution_time_ms: How long it took
    - charts: List of saved chart file paths
    """
    
    name: str = "python_code_executor"
    description: str = (
        "Executes Python code safely and returns results. "
        "Use this to write and execute data analysis code. "
        "Always import libraries inside the code. "
        "Return results as JSON for integration with other agents."
    )
    
    timeout_seconds: int = Field(default=30)
    output_dir: str = Field(default="./execution_outputs")
    execution_state: dict = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        Path(self.output_dir).mkdir(exist_ok=True)
        self.execution_state = {}
    
    def _run(self, code: str) -> str:
        """Execute Python code and return results as JSON."""
        
        execution_id = int(time.time() * 1000)
        result = {
            "execution_id": execution_id,
            "success": False,
            "code": code[:200] + "..." if len(code) > 200 else code,
            "stdout": "",
            "stderr": "",
            "execution_time_ms": 0,
            "charts": [],
            "error": None
        }
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                wrapped_code = self._wrap_code(code, temp_dir, execution_id)
                code_file = Path(temp_dir) / "exec_code.py"
                code_file.write_text(wrapped_code)
                
                start_time = time.time()
                
                proc = subprocess.run(
                    ["python", str(code_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=temp_dir,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONDONTWRITEBYTECODE": "1"
                    }
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                result["stdout"] = proc.stdout
                result["stderr"] = proc.stderr
                result["execution_time_ms"] = elapsed_ms
                
                if proc.returncode == 0:
                    result["success"] = True
                    result["charts"] = self._collect_charts(temp_dir)
                    self.execution_state[execution_id] = {
                        "output": proc.stdout,
                        "charts": result["charts"]
                    }
                else:
                    error_msg = proc.stderr if proc.stderr else f"Exit code: {proc.returncode}"
                    result["error"] = error_msg
                    
        except subprocess.TimeoutExpired:
            result["error"] = f"Execution timeout after {self.timeout_seconds}s"
        except Exception as e:
            result["error"] = f"Execution failed: {str(e)}"
        
        return json.dumps(result)
    
    def _wrap_code(self, code: str, temp_dir: str, exec_id: int) -> str:
        """Wrap code with safety and output capture."""
        # Make path safe on Windows by using forward slashes
        temp_dir_safe = temp_dir.replace('\\', '/')

        return f'''
import sys
import warnings
warnings.filterwarnings("ignore")

# Setup matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global charts list
_CHARTS = []

# ===== USER CODE STARTS =====
{code}
# ===== USER CODE ENDS =====

# Save all matplotlib figures
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    chart_path = "{temp_dir_safe}/chart_{exec_id}_{{fig_num}}.png"
    fig.savefig(chart_path, dpi=100, bbox_inches='tight')
    _CHARTS.append(chart_path)
    plt.close(fig)

if _CHARTS:
    print(f"\\n__CHARTS_GENERATED__: {{_CHARTS}}")
'''
    
    def _collect_charts(self, temp_dir: str) -> List[str]:
        """Collect and move generated charts to output directory."""
        charts = []
        for png_file in Path(temp_dir).glob("chart_*.png"):
            try:
                dest = Path(self.output_dir) / png_file.name
                dest.write_bytes(png_file.read_bytes())
                charts.append(str(dest))
            except Exception as e:
                print(f"Warning: Could not save chart {png_file}: {e}")
        return charts


# ============================================================================
# PART 2: AGENT DEFINITIONS
# ============================================================================

def create_agents(executor_tool: PythonCodeExecutorTool) -> Dict[str, Agent]:
    """Create specialized agents for data analysis pipeline."""
    
    from crewai import LLM
    
    gemini_llm = LLM(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    agents = {
        "library_import": Agent(
            role="Python Environment Setup Specialist",
            goal="Set up necessary libraries for data analysis",
            backstory=(
                "You import and prepare Python data science libraries "
                "and confirm the environment and dataset can be loaded."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "data_loading": Agent(
            role="Data Loading Specialist",
            goal="Load and validate the dataset",
            backstory=(
                "You load the dataset and provide a concise structural summary."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "data_inspection": Agent(
            role="Data Inspection Analyst",
            goal="Inspect data structure and quality",
            backstory=(
                "You examine columns, types, duplicates, and basic quality issues."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "data_cleaning": Agent(
            role="Data Cleaning Specialist",
            goal="Clean and prepare data for analysis",
            backstory=(
                "You handle missing values, duplicates, outliers, and type fixes, "
                "and report operations concisely."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "data_transformation": Agent(
            role="Feature Engineering Expert",
            goal="Transform and derive meaningful features",
            backstory=(
                "You perform feature engineering and transformations needed for analysis."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "eda_analyst": Agent(
            role="Exploratory Data Analysis Specialist",
            goal="Perform compact EDA",
            backstory=(
                "You compute key descriptive statistics and highlight main patterns "
                "in a concise, structured way."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "visualization_expert": Agent(
            role="Data Visualization Specialist",
            goal="Create informative visualizations",
            backstory=(
                "You create a small set of core charts and print only file paths and short labels."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "statistical_analyst": Agent(
            role="Statistical Analysis Expert",
            goal="Perform key statistical tests",
            backstory=(
                "You run a few relevant statistical tests and provide brief results."
            ),
            llm=gemini_llm,
            tools=[executor_tool],
            verbose=True
        ),
        
        "report_generator": Agent(
            role="Technical Report Writer",
            goal="Generate concise markdown report",
            backstory=(
                "You synthesize all previous outputs into a short, structured markdown report."
            ),
            llm=gemini_llm,
            tools=[],
            verbose=True
        )
    }
    
    return agents


# ============================================================================
# PART 3: TASK DEFINITIONS (TOKEN-OPTIMIZED)
# ============================================================================

def create_tasks(agents: Dict[str, Agent], dataset_path: str) -> Dict[str, Task]:
    """Create sequential analysis tasks."""
    
    tasks = {
        "library_import": Task(
            description=(
                f"Write Python code to:\n"
                f"- Import pandas, numpy, matplotlib, seaborn, scipy, scikit-learn.\n"
                f"- Set matplotlib to non-interactive backend.\n"
                f"- Load dataset from: {dataset_path} into a DataFrame named df.\n"
                f"- Print only a short confirmation message and versions.\n"
                f"Keep text output under ~200 tokens."
            ),
            expected_output="Short confirmation and library versions.",
            agent=agents["library_import"],
            async_execution=False
        ),
        
        "data_loading": Task(
            description=(
                f"Use Python to load the dataset from {dataset_path} (if not already loaded).\n"
                f"Print ONLY:\n"
                f"- Shape (rows, columns).\n"
                f"- Column names with dtypes.\n"
                f"- Missing value count per column.\n"
                f"- First 3 rows as a compact table.\n"
                f"Keep output under ~400 tokens."
            ),
            expected_output="Concise loading summary.",
            agent=agents["data_loading"],
            context=[],
            async_execution=False
        ),
        
        "data_inspection": Task(
            description=(
                "Inspect the loaded dataset and print a compact summary:\n"
                "- List of columns with dtype and number of unique values.\n"
                "- Number of duplicate rows.\n"
                "- Basic range (min, max) for numeric columns.\n"
                "- Brief notes on obvious quality issues.\n"
                "Avoid printing full value lists; keep output under ~500 tokens."
            ),
            expected_output="Compact inspection report.",
            agent=agents["data_inspection"],
            async_execution=False
        ),
        
        "data_cleaning": Task(
            description=(
                "Clean the dataset using Python code:\n"
                "1. Handle missing values (drop or fill based on simple strategy).\n"
                "2. Remove exact duplicate rows.\n"
                "3. Identify and optionally cap/extreme outliers.\n"
                "4. Fix obvious dtype issues.\n"
                "5. Drop rows with >50% missing values.\n"
                "Print a SHORT summary: operations performed and new shape."
            ),
            expected_output="Short cleaning report and new shape.",
            agent=agents["data_cleaning"],
            async_execution=False
        ),
        
        "data_transformation": Task(
            description=(
                "Transform the cleaned dataset:\n"
                "1. Apply basic scaling to numeric features if needed.\n"
                "2. Encode categorical variables with simple encodings.\n"
                "3. Create 1–3 useful derived features.\n"
                "Print only a summary of transformations and final dtypes.\n"
                "Avoid large tables; keep under ~400 tokens."
            ),
            expected_output="Transformation summary.",
            agent=agents["data_transformation"],
            async_execution=False
        ),
        
        "eda_analysis": Task(
            description=(
                "Perform concise EDA and print a structured summary:\n"
                "- Key descriptive stats (mean, std, min, max) for main numeric columns.\n"
                "- 2–3 notable correlations or relationships.\n"
                "- 2–3 main patterns or anomalies.\n"
                "Return this as markdown bullet points or a small JSON-like block, "
                "not exceeding ~500 tokens."
            ),
            expected_output="Compact EDA findings.",
            agent=agents["eda_analyst"],
            async_execution=False
        ),
        
        "visualizations": Task(
            description=(
                "Create a small set of core visualizations using matplotlib/seaborn:\n"
                "1. 1–2 distribution plots for key numeric variables.\n"
                "2. 1 correlation heatmap (if applicable).\n"
                "3. 1 box plot for outlier visualization.\n"
                "Save all figures with plt.savefig() and close them.\n"
                "Print ONLY a short list: chart_name -> file_path."
            ),
            expected_output="List of generated chart file paths.",
            agent=agents["visualization_expert"],
            async_execution=False
        ),
        
        "statistical_tests": Task(
            description=(
                "Perform a few basic statistical analyses:\n"
                "- 1 normality test on a main numeric column.\n"
                "- 1 correlation significance test between two key numeric columns.\n"
                "- If a suitable categorical variable exists, 1 group comparison test.\n"
                "Print a SHORT markdown section with:\n"
                "- Test name.\n"
                "- Variables.\n"
                "- p-value.\n"
                "- 1-sentence interpretation.\n"
                "Keep under ~400 tokens."
            ),
            expected_output="Concise statistical test results.",
            agent=agents["statistical_analyst"],
            async_execution=False
        )
    }
    
    # Context dependencies
    if "data_loading" in tasks:
        tasks["data_loading"].context = [tasks["library_import"]]
    if "data_inspection" in tasks:
        tasks["data_inspection"].context = [tasks["data_loading"]]
    if "data_cleaning" in tasks:
        tasks["data_cleaning"].context = [tasks["data_inspection"]]
    if "data_transformation" in tasks:
        tasks["data_transformation"].context = [tasks["data_cleaning"]]
    if "eda_analysis" in tasks:
        tasks["eda_analysis"].context = [tasks["data_transformation"]]
    if "visualizations" in tasks:
        tasks["visualizations"].context = [tasks["eda_analysis"]]
    if "statistical_tests" in tasks:
        tasks["statistical_tests"].context = [tasks["visualizations"]]
    
    return tasks


# ============================================================================
# PART 4: WORKFLOW ORCHESTRATION (1 RPS + MARKDOWN REPORT)
# ============================================================================

class DataAnalysisWorkflow:
    """Main workflow controller for sequential data analysis pipeline."""
    
    def __init__(self, dataset_path: str, output_dir: str = "./analysis_results"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.executor = PythonCodeExecutorTool(output_dir=str(self.output_dir / "charts"))
        self.agents = create_agents(self.executor)
        self.tasks = create_tasks(self.agents, dataset_path)
        
        self.results = {}
        self.charts = []
        
    def run_sequential_pipeline(self) -> Dict[str, Any]:
        """Run the sequential data analysis pipeline with rate limiting (1 RPS)."""
        
        print(f"\n{'='*70}")
        print("STARTING SEQUENTIAL DATA ANALYSIS PIPELINE")
        print(f"Dataset: {self.dataset_path}")
        print(f"Output Directory: {self.output_dir}")
        print("Rate Limiting: ~1 Gemini request per second")
        print(f"{'='*70}\n")
        
        # ----- PHASE 1: PREPARATION -----
        print("\n" + "="*70)
        print("PHASE 1: SEQUENTIAL DATA PREPARATION PIPELINE")
        print("="*70)
        
        prep_agents = [
            self.agents["library_import"],
            self.agents["data_loading"],
            self.agents["data_inspection"],
            self.agents["data_cleaning"],
            self.agents["data_transformation"]
        ]
        
        prep_tasks = [
            self.tasks["library_import"],
            self.tasks["data_loading"],
            self.tasks["data_inspection"],
            self.tasks["data_cleaning"],
            self.tasks["data_transformation"]
        ]
        
        try:
            print("\n[PHASE 1] Starting preparation pipeline...")
            for i, (agent, task) in enumerate(zip(prep_agents, prep_tasks), 1):
                print(f"\n[PHASE 1 - Task {i}/5] {agent.role}")
                print(f"[TIME] {datetime.now().strftime('%H:%M:%S')} - Starting task...")
                
                single_crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )
                
                task_result = single_crew.kickoff(inputs={"dataset_path": self.dataset_path})
                self.results[f"task_{i}"] = str(task_result)
                
                print(f"[PHASE 1 - Task {i}/5] ✓ Completed")
                
                if i < len(prep_tasks):
                    print("[RATE LIMITING] Waiting 15 seconds before next task...")
                    time.sleep(15)
            
            self.results["preparation"] = "\n---\n".join(
                self.results.get(f"task_{i}", "") for i in range(1, len(prep_tasks) + 1)
            )
            print("\n[PHASE 1] ✓ Preparation phase completed")
        
        except Exception as e:
            print(f"\n[PHASE 1] ✗ Error in preparation phase: {e}")
            self.results["preparation"] = f"Error: {str(e)}"
            return self.results
        
        print("\n[RATE LIMITING] Waiting 15 seconds between phases...")
        time.sleep(15)
        
        # ----- PHASE 2: ANALYSIS -----
        print("\n" + "="*70)
        print("PHASE 2: SEQUENTIAL ANALYSIS GROUP")
        print("="*70)
        
        analysis_agents = [
            self.agents["eda_analyst"],
            self.agents["visualization_expert"],
            self.agents["statistical_analyst"]
        ]
        
        analysis_tasks = [
            self.tasks["eda_analysis"],
            self.tasks["visualizations"],
            self.tasks["statistical_tests"]
        ]
        
        try:
            print("\n[PHASE 2] Starting analysis phase...")
            for i, (agent, task) in enumerate(zip(analysis_agents, analysis_tasks), 1):
                print(f"\n[PHASE 2 - Task {i}/3] {agent.role}")
                print(f"[TIME] {datetime.now().strftime('%H:%M:%S')} - Starting task...")
                
                single_crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )
                
                task_result = single_crew.kickoff(inputs={"context": self.results["preparation"]})
                self.results[f"analysis_task_{i}"] = str(task_result)
                
                print(f"[PHASE 2 - Task {i}/3] ✓ Completed")
                
                if i < len(analysis_tasks):
                    print("[RATE LIMITING] Waiting 15 seconds before next task...")
                    time.sleep(15)
            
            self.results["analysis"] = "\n---\n".join(
                self.results.get(f"analysis_task_{i}", "") for i in range(1, len(analysis_tasks) + 1)
            )
            print("\n[PHASE 2] ✓ Analysis phase completed")
        
        except Exception as e:
            print(f"\n[PHASE 2] ✗ Error in analysis phase: {e}")
            self.results["analysis"] = f"Error: {str(e)}"
        
        print("\n[RATE LIMITING] Waiting 10 seconds before final report...")
        time.sleep(10)
        
        # ----- PHASE 3: REPORT GENERATION (MARKDOWN) -----
        print("\n" + "="*70)
        print("PHASE 3: REPORT GENERATION (MARKDOWN)")
        print("="*70)
        
        report_task = Task(
            description=(
                "Using the following analysis context, write a concise markdown report.\n\n"
                "=== PREPARATION PHASE ===\n"
                f"{self.results['preparation']}\n\n"
                "=== ANALYSIS PHASE ===\n"
                f"{self.results.get('analysis', 'N/A')}\n\n"
                "Requirements:\n"
                "- Output MUST be valid markdown.\n"
                "- Max length about 800–1000 words.\n"
                "- Use sections: # Executive Summary, ## Data, ## Cleaning, "
                "## EDA, ## Statistics, ## Key Insights.\n"
                "- Use short bullet points, avoid repeating large tables verbatim."
            ),
            expected_output="Concise markdown report (<= ~1000 words).",
            agent=self.agents["report_generator"]
        )
        
        try:
            print("\n[PHASE 3] Starting report generation...")
            print(f"[TIME] {datetime.now().strftime('%H:%M:%S')} - Starting report task...")
            report_result = report_task.execute_sync(agent=self.agents["report_generator"])
            self.results["report"] = str(report_result)
            print("\n[PHASE 3] ✓ Report generation completed")
        except Exception as e:
            print(f"\n[PHASE 3] ✗ Error in report generation: {e}")
            self.results["report"] = f"Error: {str(e)}"
        
        self.charts = list((self.output_dir / "charts").glob("*.png"))
        
        return self.results
    
    def generate_markdown_report(self) -> str:
        """Save the LLM-generated markdown report to a .md file."""
        report_md = self.results.get("report", "")
        if not report_md:
            report_md = "# Analysis Report\n\n_No report content generated._"
        
        report_path = self.output_dir / "analysis_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        
        print(f"\n{'='*70}")
        print("✓ MARKDOWN REPORT GENERATED")
        print(f"Location: {report_path}")
        print(f"{'='*70}\n")
        
        return str(report_path)
    
    # Backward-compatible alias if run.py still calls generate_html_report
    def generate_html_report(self) -> str:
        """Compatibility wrapper: generates markdown instead of HTML."""
        return self.generate_markdown_report()
    
    def _embed_charts(self) -> str:
        """No-op placeholder kept for compatibility (not used in markdown)."""
        return ""


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    
    dataset_path = "sample_data.csv"
    
    if not Path(dataset_path).exists():
        print(f"Creating sample dataset: {dataset_path}")
        create_sample_dataset(dataset_path)
    
    workflow = DataAnalysisWorkflow(
        dataset_path=dataset_path,
        output_dir="./analysis_results"
    )
    
    results = workflow.run_sequential_pipeline()
    
    report_path = workflow.generate_markdown_report()
    
    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"Charts saved to: {workflow.output_dir / 'charts'}")
    
    return report_path


def create_sample_dataset(path: str):
    """Create a sample dataset for testing."""
    
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Income': np.random.normal(50000, 20000, n_samples).astype(int),
        'Experience_Years': np.random.randint(0, 40, n_samples),
        'Score': np.random.normal(75, 15, n_samples),
        'Department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
        'Satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"✓ Sample dataset created: {path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
