# ============================================================================
# COMPLETE CREWAI DATA ANALYSIS WORKFLOW
# Multi-Agent Sequential Pipeline for Gemini (Token-Optimized)
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
        temp_dir_safe = temp_dir.replace('\\', '/')

        return f'''
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_CHARTS = []

# ===== USER CODE STARTS =====
{code}
# ===== USER CODE ENDS =====

for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    chart_path = "{temp_dir_safe}/chart_{exec_id}_{{fig_num}}.png"
    fig.savefig(chart_path, dpi=100, bbox_inches='tight')
    _CHARTS.append(chart_path)
    plt.close(fig)

if _CHARTS:
    print("\\n__CHARTS_GENERATED__:", _CHARTS)
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
# PART 2: AGENT DEFINITIONS (TOKEN-OPTIMIZED CONFIG)
# ============================================================================

def _make_gemini_llm(max_output_tokens: int, thinking_budget: int = 0):
    """
    Create a token-capped Gemini LLM instance.
    thinking_budget=0 disables expensive 'thinking' tokens for cheap calls.
    """
    from crewai import LLM

    generation_config = {
        "max_output_tokens": max_output_tokens,
    }

    if thinking_budget > 0:
        generation_config["thinking"] = {"budget_tokens": thinking_budget}

    return LLM(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        config=generation_config,
    )

def create_agents(executor_tool: PythonCodeExecutorTool) -> Dict[str, Agent]:
    """Create specialized agents for data analysis pipeline."""

    llm_short = _make_gemini_llm(max_output_tokens=350, thinking_budget=0)
    llm_medium = _make_gemini_llm(max_output_tokens=800, thinking_budget=512)
    llm_long = _make_gemini_llm(max_output_tokens=1600, thinking_budget=1024)

    agents = {
        "library_import": Agent(
            role="Python Environment Setup Specialist",
            goal="Quickly prepare core data science libraries and confirm the dataset loads.",
            backstory=(
                "You focus on minimal, correct import and load code. "
                "You output compact Python only, with tiny confirmations."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_loading": Agent(
            role="Data Loading Specialist",
            goal="Load the dataset and summarise its basic structure.",
            backstory=(
                "You only report shape, column names, dtypes, and missing counts. "
                "You avoid long tables or prose."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_inspection": Agent(
            role="Data Inspection Analyst",
            goal="Inspect schema and quality issues in a compact way.",
            backstory=(
                "You highlight duplicates, ranges, and obvious issues with short bullet points."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_cleaning": Agent(
            role="Data Cleaning Specialist",
            goal="Apply simple, transparent cleaning steps and report new shape.",
            backstory=(
                "You perform only essential cleaning (missing values, duplicates, outliers, dtypes) "
                "and summarise actions briefly."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_transformation": Agent(
            role="Feature Engineering Expert",
            goal="Create a small set of clear, useful feature transformations.",
            backstory=(
                "You apply light scaling/encoding and a few derived features, then list them quickly."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "eda_analyst": Agent(
            role="Exploratory Data Analysis Specialist",
            goal="Provide concise EDA with only key statistics and patterns.",
            backstory=(
                "You avoid long narratives and focus on the 2–3 most important findings."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "visualization_expert": Agent(
            role="Data Visualization Specialist",
            goal="Generate a few core charts and list their file paths.",
            backstory=(
                "You create only essential plots and output chart_name → file_path pairs."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "statistical_analyst": Agent(
            role="Statistical Analysis Expert",
            goal="Run a few key tests and summarise results briefly.",
            backstory=(
                "You report test name, variables, p-value, and a one-line interpretation only."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "report_generator": Agent(
            role="Technical Report Writer",
            goal="Produce a compact markdown report that stitches all steps together.",
            backstory=(
                "You summarise previous outputs into a short, structured markdown report, "
                "avoiding repetition and large tables."
            ),
            llm=llm_long,
            tools=[],
            verbose=True,
        ),
    }

    return agents

# ============================================================================
# PART 3: TASK DEFINITIONS (TIGHTER TOKEN BUDGETS)
# ============================================================================

def create_tasks(agents: Dict[str, Agent], dataset_path: str) -> Dict[str, Task]:
    """Create sequential analysis tasks."""

    tasks = {
        "library_import": Task(
            description=(
                f"Write Python code to:\n"
                f"- Import pandas, numpy, matplotlib, seaborn, scipy, scikit-learn.\n"
                f"- Set matplotlib to non-interactive backend.\n"
                f"- Load dataset from: '{dataset_path}' into df using pd.read_csv('{dataset_path}').\n"
                f"- Print a one-line confirmation and library versions.\n"
                f"Output: code only (no explanation) plus minimal print statements.\n"
                f"IMPORTANT: Do NOT hard-code any different file name or path."
            ),
            expected_output="Short Python snippet that imports libraries and loads df.",
            agent=agents["library_import"],
            async_execution=False,
        ),

        "data_loading": Task(
            description=(
                f"Use Python to ensure the dataset at '{dataset_path}' is loaded in df.\n"
                f"Always load using: df = pd.read_csv('{dataset_path}').\n"
                f"Print ONLY:\n"
                f"- Shape (rows, columns).\n"
                f"- Column names with dtypes.\n"
                f"- Missing value count per column.\n"
                f"- Head(3) as a compact table.\n"
                f"Keep text highly compact; avoid extra commentary."
            ),
            expected_output="Concise loading summary with shape, dtypes, missing values and 3-row preview.",
            agent=agents["data_loading"],
            context=[],
            async_execution=False,
        ),

        "data_inspection": Task(
            description=(
                f"First, load the dataset: df = pd.read_csv('{dataset_path}')\n"
                f"Then inspect it and print a small summary:\n"
                f"- Columns with dtype and number of unique values.\n"
                f"- Count of duplicate rows.\n"
                f"- Min and max for numeric columns.\n"
                f"- 2–3 bullet points on obvious quality issues.\n"
                f"Do not print full distributions or long tables."
            ),
            expected_output="Short inspection report with bullets.",
            agent=agents["data_inspection"],
            async_execution=False,
        ),

        "data_cleaning": Task(
            description=(
                f"First, load the dataset: df = pd.read_csv('{dataset_path}')\n"
                f"Then clean it using Python code:\n"
                f"1. Handle missing values with simple, documented strategy.\n"
                f"2. Drop exact duplicate rows.\n"
                f"3. Optionally cap extreme numeric outliers.\n"
                f"4. Fix obvious dtype issues.\n"
                f"5. Drop rows with >50% missing.\n"
                f"Print a brief summary: operations performed and new shape."
            ),
            expected_output="Short cleaning report listing steps and resulting shape.",
            agent=agents["data_cleaning"],
            async_execution=False,
        ),

        "data_transformation": Task(
            description=(
                f"First, load the dataset: df = pd.read_csv('{dataset_path}')\n"
                f"Then transform it:\n"
                f"1. Scale or normalise key numeric features if helpful.\n"
                f"2. Encode categorical variables with simple encodings.\n"
                f"3. Create 1–3 meaningful derived features.\n"
                f"Print only a compact list of transformations and final dtypes."
            ),
            expected_output="Transformation summary with feature list and dtypes.",
            agent=agents["data_transformation"],
            async_execution=False,
        ),

        "eda_analysis": Task(
            description=(
                f"First, load the dataset: df = pd.read_csv('{dataset_path}')\n"
                f"Then perform concise EDA and print a structured summary:\n"
                f"- Descriptive stats (mean, std, min, max) for key numeric columns.\n"
                f"- 2–3 notable correlations or relationships.\n"
                f"- 2–3 main patterns or anomalies.\n"
                f"Return as markdown bullets or a small JSON-style block; keep it compact."
            ),
            expected_output="Compact EDA findings in bullet form.",
            agent=agents["eda_analyst"],
            async_execution=False,
        ),

        "visualizations": Task(
            description=(
                f"First, load the dataset: df = pd.read_csv('{dataset_path}')\n"
                f"Then create a few core visualisations using matplotlib/seaborn:\n"
                f"1. 1–2 distribution plots for important numeric variables.\n"
                f"2. 1 correlation heatmap (if applicable).\n"
                f"3. 1 box plot for outliers.\n"
                f"Save all figures with plt.savefig() and close them.\n"
                f"Print ONLY a short list: chart_name -> file_path."
            ),
            expected_output="List of generated chart file paths with short labels.",
            agent=agents["visualization_expert"],
            async_execution=False,
        ),

        "statistical_tests": Task(
            description=(
                f"First, load the dataset: df = pd.read_csv('{dataset_path}')\n"
                f"Then run a few basic statistical analyses:\n"
                f"- One normality test on a main numeric column.\n"
                f"- One correlation significance test between two key numeric columns.\n"
                f"- If possible, one simple group comparison using a categorical variable.\n"
                f"Print a SHORT markdown section with:\n"
                f"- Test name\n"
                f"- Variables\n"
                f"- p-value\n"
                f"- 1-sentence interpretation."
            ),
            expected_output="Concise statistical test results.",
            agent=agents["statistical_analyst"],
            async_execution=False,
        ),
    }

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
# PART 4: WORKFLOW ORCHESTRATION
# ============================================================================

class DataAnalysisWorkflow:
    """Main workflow controller for sequential data analysis pipeline."""

    def __init__(self, dataset_path: str, output_dir: str = "./analysis_results"):
        # Convert to absolute paths so subprocess can find files from any working directory
        self.dataset_path = str(Path(dataset_path).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)

        self.executor = PythonCodeExecutorTool(output_dir=str(self.output_dir / "charts"))
        self.agents = create_agents(self.executor)
        self.tasks = create_tasks(self.agents, self.dataset_path)

        self.results = {}
        self.charts = []

    def run_sequential_pipeline(self) -> Dict[str, Any]:
        """Run the sequential data analysis pipeline."""

        print(f"\n{'='*70}")
        print("STARTING SEQUENTIAL DATA ANALYSIS PIPELINE")
        print(f"Dataset: {self.dataset_path}")
        print(f"Output Directory: {self.output_dir}")
        print("LLM: Gemini 2.5 Flash (token-optimized config)")
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
            self.agents["data_transformation"],
        ]

        prep_tasks = [
            self.tasks["library_import"],
            self.tasks["data_loading"],
            self.tasks["data_inspection"],
            self.tasks["data_cleaning"],
            self.tasks["data_transformation"],
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
                    verbose=True,
                )

                task_result = single_crew.kickoff(inputs={"dataset_path": self.dataset_path})
                self.results[f"task_{i}"] = str(task_result)

                print(f"[PHASE 1 - Task {i}/5] ✓ Completed")

            self.results["preparation"] = "\n---\n".join(
                self.results.get(f"task_{i}", "") for i in range(1, len(prep_tasks) + 1)
            )
            print("\n[PHASE 1] ✓ Preparation phase completed")

        except Exception as e:
            print(f"\n[PHASE 1] ✗ Error in preparation phase: {e}")
            self.results["preparation"] = f"Error: {str(e)}"
            return self.results

        print("\n[RATE LIMITING] Waiting 10 seconds between phases...")
        time.sleep(10)

        # ----- PHASE 2: ANALYSIS -----
        print("\n" + "="*70)
        print("PHASE 2: SEQUENTIAL ANALYSIS GROUP")
        print("="*70)

        analysis_agents = [
            self.agents["eda_analyst"],
            self.agents["visualization_expert"],
            self.agents["statistical_analyst"],
        ]

        analysis_tasks = [
            self.tasks["eda_analysis"],
            self.tasks["visualizations"],
            self.tasks["statistical_tests"],
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
                    verbose=True,
                )

                task_result = single_crew.kickoff(inputs={"context": self.results["preparation"]})
                self.results[f"analysis_task_{i}"] = str(task_result)

                print(f"[PHASE 2 - Task {i}/3] ✓ Completed")

            self.results["analysis"] = "\n---\n".join(
                self.results.get(f"analysis_task_{i}", "") for i in range(1, len(analysis_tasks) + 1)
            )
            print("\n[PHASE 2] ✓ Analysis phase completed")

        except Exception as e:
            print(f"\n[PHASE 2] ✗ Error in analysis phase: {e}")
            self.results["analysis"] = f"Error: {str(e)}"

        print("\n[RATE LIMITING] Waiting 8 seconds before final report...")
        time.sleep(8)

        # ----- PHASE 3: REPORT GENERATION (MARKDOWN) -----
        print("\n" + "="*70)
        print("PHASE 3: REPORT GENERATION (MARKDOWN)")
        print("="*70)

        report_task = Task(
            description=(
                "Using the analysis context below, write a concise markdown report.\n\n"
                "=== PREPARATION PHASE ===\n"
                f"{self.results['preparation']}\n\n"
                "=== ANALYSIS PHASE ===\n"
                f"{self.results.get('analysis', 'N/A')}\n\n"
                "Requirements:\n"
                "- Output valid markdown only.\n"
                "- Max length about 800–1000 words.\n"
                "- Sections: # Executive Summary, ## Data, ## Cleaning, "
                "## EDA, ## Statistics, ## Key Insights.\n"
                "- Use short bullet points and avoid repeating large tables."
            ),
            expected_output="Concise markdown report (<= ~1000 words).",
            agent=self.agents["report_generator"],
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

    def generate_html_report(self) -> str:
        """Compatibility wrapper: generates markdown instead of HTML."""
        return self.generate_markdown_report()

    def _embed_charts(self) -> str:
        """No-op placeholder kept for compatibility (not used in markdown)."""
        return ""

# ============================================================================
# PART 5: MAIN EXECUTION (STANDALONE TEST ONLY)
# ============================================================================

def create_sample_dataset(path: str):
    """Create a sample dataset for testing."""

    np.random.seed(42)
    n_samples = 200

    data = {
        "Age": np.random.randint(18, 80, n_samples),
        "Income": np.random.normal(50000, 20000, n_samples).astype(int),
        "Experience_Years": np.random.randint(0, 40, n_samples),
        "Score": np.random.normal(75, 15, n_samples),
        "Department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], n_samples),
        "Satisfaction": np.random.choice([1, 2, 3, 4, 5], n_samples),
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"✓ Sample dataset created: {path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

def main():
    """Standalone test entry point (not used when imported from run.py)."""
    dataset_path = "sample_data.csv"
    if not Path(dataset_path).exists():
        print(f"Creating sample dataset: {dataset_path}")
        create_sample_dataset(dataset_path)

    workflow = DataAnalysisWorkflow(
        dataset_path=dataset_path,
        output_dir="./analysis_results",
    )

    results = workflow.run_sequential_pipeline()
    report_path = workflow.generate_markdown_report()

    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"Charts saved to: {workflow.output_dir / 'charts'}")

    return report_path

if __name__ == "__main__":
    main()
