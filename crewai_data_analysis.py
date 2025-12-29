# ============================================================================
# COMPLETE CREWAI DATA ANALYSIS WORKFLOW (STATEFUL, TOKEN-OPTIMIZED)
# ============================================================================

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import Field

import google.generativeai as genai  # noqa: F401


# ============================================================================
# PART 1: STATEFUL PYTHON SESSION TOOL
# ============================================================================

class PythonSessionTool(BaseTool):
    """
    Long-lived, in-process Python execution environment.

    Key properties:
    - Single interpreter, shared globals between calls.
    - Dataset loaded once into df_raw, later steps use df_clean, df_features, etc.
    - Lightweight state validation helpers.
    - Returns structured JSON: stdout, success, charts, error, state_flags.
    """

    name: str = "python_stateful_executor"
    description: str = (
        "Executes Python code in a shared in-memory session that persists state "
        "between calls. Use only the existing variables (df_raw, df_clean, "
        "df_features, validation_report, etc.) and DO NOT reload the CSV."
    )

    output_dir: str = Field(default="./execution_outputs")
    session_globals: dict = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._init_base_session()

    # ------------------------------------------------------------------ #
    # Session & state helpers
    # ------------------------------------------------------------------ #
    def _init_base_session(self):
        """Initialize the shared session globals once."""
        base_globals = {
            "__name__": "__session__",
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "matplotlib": matplotlib,
        }
        self.session_globals = base_globals

    def init_session(self, dataset_path: str):
        """
        Called once from workflow before any LLM tool calls.

        - Loads CSV into df_raw
        - Sets df_clean and df_features to None initially
        """
        code = f"""
import pandas as _pd

dataset_path = r\"\"\"{dataset_path}\"\"\"
df_raw = _pd.read_csv(dataset_path)
df_clean = None
df_features = None
validation_report = {{}}
"""
        exec(code, self.session_globals)

    def validate_state(self) -> Dict[str, bool]:
        """Check presence of key objects between tasks."""
        g = self.session_globals
        flags = {
            "has_df_raw": "df_raw" in g and isinstance(g.get("df_raw"), pd.DataFrame),
            "has_df_clean": "df_clean" in g and (
                g.get("df_clean") is None or isinstance(g.get("df_clean"), pd.DataFrame)
            ),
            "has_df_features": "df_features" in g and (
                g.get("df_features") is None or isinstance(g.get("df_features"), pd.DataFrame)
            ),
            "has_validation_report": "validation_report" in g,
        }
        return flags

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def _run(self, code: str) -> str:
        """
        Execute Python code inside the shared session and return JSON.

        The code can assume:
        - df_raw: original dataset (never modified after load)
        - df_clean: cleaned dataset (may start as None)
        - df_features: transformed dataset/features (may start as None)
        - validation_report: dict with validation results
        """
        result = {
            "success": False,
            "stdout": "",
            "error": None,
            "charts": [],
            "state_flags": {},
        }

        # Capture prints in-memory
        import io
        import contextlib

        stdout_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, self.session_globals)
            result["success"] = True
        except Exception as e:
            result["error"] = repr(e)
            print("EXECUTOR ERROR:", repr(e))

        result["stdout"] = stdout_buffer.getvalue()
        stdout_buffer.close()

        # Collect charts created with plt
        charts = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            chart_path = Path(self.output_dir) / f"chart_{int(time.time() * 1000)}_{fig_num}.png"
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            charts.append(str(chart_path))
            plt.close(fig)
        result["charts"] = charts

        # Attach state validation flags
        result["state_flags"] = self.validate_state()

        return json.dumps(result)


# ============================================================================
# PART 2: LLM CONFIG
# ============================================================================

def _make_gemini_llm(max_output_tokens: int, thinking_budget: int = 0):
    from crewai import LLM

    generation_config = {"max_output_tokens": max_output_tokens}
    if thinking_budget > 0:
        generation_config["thinking"] = {"budget_tokens": thinking_budget}

    return LLM(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        config=generation_config,
    )


# ============================================================================
# PART 3: AGENT DEFINITIONS (STATE-AWARE)
# ============================================================================

def create_agents(executor_tool: PythonSessionTool) -> Dict[str, Agent]:
    llm_short = _make_gemini_llm(max_output_tokens=320, thinking_budget=0)
    llm_medium = _make_gemini_llm(max_output_tokens=640, thinking_budget=256)
    llm_long = _make_gemini_llm(max_output_tokens=1200, thinking_budget=512)

    agents = {
        "library_import": Agent(
            role="Python Environment Setup Specialist",
            goal="Prepare the Python environment and confirm df_raw is loaded and accessible.",
            backstory=(
                "You work with a persistent in-memory Python session. "
                "You never reload the CSV file. You just verify df_raw and environment."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_loading": Agent(
            role="Data Structure Summariser",
            goal="Summarise the structure of df_raw already in memory.",
            backstory=(
                "You assume df_raw already exists in the shared session. "
                "You do not call read_csv. You only inspect df_raw."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_inspection": Agent(
            role="Data Inspection Analyst",
            goal="Inspect df_raw and highlight schema and quality issues.",
            backstory=(
                "You use df_raw from the shared session and never reload the dataset. "
                "You emit concise bullet-point diagnostics."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_validation": Agent(
            role="Data Validation Specialist",
            goal="Run core data validation rules on df_raw and store results in validation_report.",
            backstory=(
                "You implement simple, explicit validation rules on df_raw and record them "
                "in a validation_report dict so later tasks can check them."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_cleaning": Agent(
            role="Data Cleaning Specialist",
            goal="Create df_clean from df_raw using minimal, transparent cleaning steps.",
            backstory=(
                "You read df_raw, create df_clean, document operations via prints, "
                "and never reload the CSV. You respect validation_report findings."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_transformation": Agent(
            role="Feature Engineering Expert",
            goal="Create df_features from df_clean and list new features.",
            backstory=(
                "You assume df_clean exists, derive df_features, and keep transformations simple. "
                "No re-loading from disk."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "eda_analysis": Agent(
            role="Exploratory Data Analysis Specialist",
            goal="Provide concise EDA using df_features if available, else df_clean or df_raw.",
            backstory=(
                "You prioritise df_features, then df_clean, then df_raw, in that order. "
                "You keep outputs short and structured."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "visualizations": Agent(
            role="Data Visualization Specialist",
            goal="Generate a small set of core charts from df_features/df_clean.",
            backstory=(
                "You use the shared session's DataFrames and save figures only via plt.savefig. "
                "You output chart_name → file_path pairs."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "statistical_tests": Agent(
            role="Statistical Analysis Expert",
            goal="Run a few key tests on df_features/df_clean.",
            backstory=(
                "You keep tests simple and use the data already in memory. "
                "No disk I/O and no CSV reloading."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "report_generator": Agent(
            role="Technical Report Writer",
            goal="Produce a compact markdown report summarising the whole pipeline.",
            backstory=(
                "You stitch together previous step outputs into a concise markdown report. "
                "You assume all computations are already done."
            ),
            llm=llm_long,
            tools=[],
            verbose=True,
        ),
    }

    return agents


# ============================================================================
# PART 4: TASK DEFINITIONS (NO RELOADING, STATEFUL)
# ============================================================================

def create_tasks(agents: Dict[str, Agent]) -> Dict[str, Task]:
    tasks = {
        "library_import": Task(
            description=(
                "Use the shared Python session (already created by the tool).\n"
                "- Confirm that pandas, numpy, matplotlib, seaborn are imported.\n"
                "- Confirm that df_raw exists and print its shape only.\n"
                "- Do NOT call read_csv or touch the filesystem.\n"
                "Return: very short Python code that only inspects existing df_raw."
            ),
            expected_output="Short Python snippet that checks libraries and df_raw shape.",
            agent=agents["library_import"],
            async_execution=False,
        ),
        "data_loading": Task(
            description=(
                "Using the in-memory df_raw (DO NOT reload from disk):\n"
                "- Print shape (rows, columns).\n"
                "- Print column names with dtypes.\n"
                "- Print missing value count per column.\n"
                "- Print df_raw.head(3).\n"
                "Return only concise Python code plus minimal print statements."
            ),
            expected_output="Concise summary of df_raw structure.",
            agent=agents["data_loading"],
            async_execution=False,
        ),
        "data_inspection": Task(
            description=(
                "Work with the existing df_raw in memory.\n"
                "Write ONE short Python script that:\n"
                "- Prints a table: column name, dtype, number of unique values using df_raw.dtypes and nunique().\n"
                "- Prints count of duplicate rows using df_raw.duplicated().sum().\n"
                "- Prints min and max for numeric columns using df_raw.describe().loc[['min','max']].\n"
                "- Prints 2 or 3 bullet-style lines describing obvious quality issues based ONLY on these stats.\n"
                "Do NOT use loops over rows, do NOT reload the dataset, and keep output very small."
            ),
            expected_output="Short inspection report of df_raw.",
            agent=agents["data_inspection"],
            async_execution=False,
        ),
        "data_validation": Task(
            description=(
                "Implement data validation rules using df_raw in memory.\n"
                "- Create/overwrite a dict called validation_report.\n"
                "- Include checks such as: missingness per column, numeric range issues, "
                "unexpected category levels, and duplicate key rows if applicable.\n"
                "- Store results in validation_report and print a concise summary.\n"
                "Do NOT reload the dataset and do NOT create new DataFrames here."
            ),
            expected_output="Short validation summary and a populated validation_report dict.",
            agent=agents["data_validation"],
            async_execution=False,
        ),
        "data_cleaning": Task(
            description=(
                "Using df_raw and validation_report from the shared session:\n"
                "- Create df_clean as a cleaned version of df_raw.\n"
                "- Handle missing values with a simple strategy and document it in prints.\n"
                "- Drop exact duplicate rows.\n"
                "- Optionally cap extreme numeric outliers.\n"
                "- Fix obvious dtype issues.\n"
                "- Drop rows with >50% missing values.\n"
                "- Print a brief summary: operations performed and df_clean.shape.\n"
                "Do not reload or save the dataset to disk."
            ),
            expected_output="Short cleaning report, df_clean created.",
            agent=agents["data_cleaning"],
            async_execution=False,
        ),
        "data_transformation": Task(
            description=(
                "Using df_clean from the shared session:\n"
                "- Create df_features as a transformed/feature-engineered DataFrame.\n"
                "- Apply light scaling/normalisation to key numeric features if useful.\n"
                "- Encode categorical variables using simple encodings.\n"
                "- Create 1–3 meaningful derived features.\n"
                "- Print a compact list of transformations and df_features.dtypes.\n"
                "No disk I/O and no CSV reloading."
            ),
            expected_output="Transformation summary with df_features defined.",
            agent=agents["data_transformation"],
            async_execution=False,
        ),
        "eda_analysis": Task(
            description=(
                "Perform concise EDA using the best available table:\n"
                "- Prefer df_features; if it is None, fall back to df_clean, then df_raw.\n"
                "- Print descriptive stats (mean, std, min, max) for key numeric columns.\n"
                "- Print 2–3 notable correlations or relationships.\n"
                "- Print 2–3 main patterns or anomalies.\n"
                "Output as markdown-style bullets in print statements; keep it compact."
            ),
            expected_output="Compact EDA findings.",
            agent=agents["eda_analysis"],
            async_execution=False,
        ),
        "visualizations": Task(
            description=(
                "Using df_features or df_clean (no disk reload):\n"
                "- Create 1–2 distribution plots for important numeric variables.\n"
                "- Create 1 correlation heatmap if numeric columns exist.\n"
                "- Create 1 box plot to show outliers.\n"
                "- Use matplotlib/seaborn, and call plt.tight_layout() before finishing.\n"
                "Figures will be saved automatically by the tool; just print a short "
                "list of chart_name -> expected_file_stub (no full path needed)."
            ),
            expected_output="List of chart labels and expected file names.",
            agent=agents["visualizations"],
            async_execution=False,
        ),
        "statistical_tests": Task(
            description=(
                "Use ONLY the DataFrames already in memory in the shared session.\n"
                "- Prefer df_features; if it is None, use df_clean; if that is None, use df_raw.\n"
                "- Do NOT call pd.read_csv or access any file paths.\n"
                "- Before using a column name, check that it exists in df.columns; if not, skip that test "
                "and print a short note.\n"
                "- Run one normality test on a numeric column (for example 'Price' if it exists).\n"
                "- Run one correlation significance test between two numeric columns that exist "
                "(for example 'Price' and 'Engine size' if both are present).\n"
                "- Run one simple group comparison using a categorical variable that exists "
                "(for example 'Fuel_Type' if present).\n"
                "Print a SHORT markdown-style section with:\n"
                "- Test name\n"
                "- Variables\n"
                "- p-value\n"
                "- One-sentence interpretation."
            ),
            expected_output="Concise statistical test results.",
            agent=agents["statistical_tests"],
            async_execution=False,
        ),
    }

    # Explicit context ordering to encourage stateful reasoning
    tasks["data_loading"].context = [tasks["library_import"]]
    tasks["data_inspection"].context = [tasks["data_loading"]]
    tasks["data_validation"].context = [tasks["data_inspection"]]
    tasks["data_cleaning"].context = [tasks["data_validation"]]
    tasks["data_transformation"].context = [tasks["data_cleaning"]]
    tasks["eda_analysis"].context = [tasks["data_transformation"]]
    tasks["visualizations"].context = [tasks["eda_analysis"]]
    tasks["statistical_tests"].context = [tasks["visualizations"]]

    return tasks


# ============================================================================
# PART 5: WORKFLOW ORCHESTRATION WITH RETRIES
# ============================================================================

class DataAnalysisWorkflow:
    """Main workflow controller for sequential data analysis pipeline."""

    def __init__(self, dataset_path: str, output_dir: str = "./analysis_results"):
        self.dataset_path = str(Path(dataset_path).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)

        # Shared, long-lived Python session tool
        self.executor = PythonSessionTool(output_dir=str(self.output_dir / "charts"))
        # Load the dataset once into df_raw
        self.executor.init_session(self.dataset_path)

        self.agents = create_agents(self.executor)
        self.tasks = create_tasks(self.agents)

        self.results: Dict[str, Any] = {}
        self.charts: List[Path] = []

    # ------------------------------------------------------------------ #
    # Simple retry wrapper
    # ------------------------------------------------------------------ #
    def _run_task_with_retry(
        self,
        agent: Agent,
        task: Task,
        task_key: str,
        max_retries: int = 2,
        delay_seconds: int = 4,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Run a Crew with a single agent+task, with basic retry on failure.

        - Retries on exceptions or explicit 'Error' in the stringified result.
        - Waits delay_seconds between attempts.
        """
        attempt = 0
        last_error = None
        extra_inputs = extra_inputs or {}

        while attempt <= max_retries:
            attempt += 1
            try:
                print(f"[RETRY] Running {task_key}, attempt {attempt}/{max_retries + 1}")
                single_crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                result = single_crew.kickoff(inputs=extra_inputs)
                result_str = str(result)
                if "Error" in result_str or "Traceback" in result_str:
                    last_error = f"Task output indicates error: {result_str[:200]}"
                    print(f"[RETRY] Detected error in {task_key}: {last_error}")
                    if attempt <= max_retries:
                        time.sleep(delay_seconds)
                        continue
                self.results[task_key] = result_str
                return result
            except Exception as e:
                last_error = repr(e)
                print(f"[RETRY] Exception in {task_key}: {last_error}")
                if attempt <= max_retries:
                    time.sleep(delay_seconds)
                    continue

        self.results[task_key] = f"Error after retries: {last_error}"
        return self.results[task_key]

    # ------------------------------------------------------------------ #
    # Main pipeline
    # ------------------------------------------------------------------ #
    def run_sequential_pipeline(self) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print("STARTING SEQUENTIAL DATA ANALYSIS PIPELINE (STATEFUL)")
        print(f"Dataset: {self.dataset_path}")
        print(f"Output Directory: {self.output_dir}")
        print("LLM: Gemini 2.5 Flash (token-optimized config)")
        print(f"{'='*70}\n")

        # ----- PHASE 1: PREPARATION -----
        print("\n" + "="*70)
        print("PHASE 1: SEQUENTIAL DATA PREPARATION PIPELINE")
        print("="*70)

        prep_order = [
            ("task_1_library_import", "library_import"),
            ("task_2_data_loading", "data_loading"),
            ("task_3_data_inspection", "data_inspection"),
            ("task_4_data_validation", "data_validation"),
            ("task_5_data_cleaning", "data_cleaning"),
            ("task_6_data_transformation", "data_transformation"),
        ]

        for i, (task_key, task_name) in enumerate(prep_order, 1):
            agent = self.agents[task_name]
            task = self.tasks[task_name]
            print(f"\n[PHASE 1 - Task {i}/{len(prep_order)}] {agent.role}")
            print(f"[TIME] {datetime.now().strftime('%H:%M:%S')} - Starting task...")

            self._run_task_with_retry(
                agent=agent,
                task=task,
                task_key=task_key,
                extra_inputs={},
            )
            print(f"[PHASE 1 - Task {i}/{len(prep_order)}] ✓ Completed (or max retries reached)")

        self.results["preparation"] = "\n---\n".join(
            self.results.get(task_key, "") for task_key, _ in prep_order
        )
        print("\n[PHASE 1] Preparation phase finished (with retries where needed)")

        print("\n[RATE LIMITING] Waiting 8 seconds between phases...")
        time.sleep(8)

        # ----- PHASE 2: ANALYSIS -----
        print("\n" + "="*70)
        print("PHASE 2: SEQUENTIAL ANALYSIS GROUP")
        print("="*70)

        analysis_order = [
            ("analysis_task_1_eda", "eda_analysis"),
            ("analysis_task_2_visualizations", "visualizations"),
            ("analysis_task_3_statistics", "statistical_tests"),
        ]

        for i, (task_key, task_name) in enumerate(analysis_order, 1):
            agent = self.agents[task_name]
            task = self.tasks[task_name]
            print(f"\n[PHASE 2 - Task {i}/{len(analysis_order)}] {agent.role}")
            print(f"[TIME] {datetime.now().strftime('%H:%M:%S')} - Starting task...")

            self._run_task_with_retry(
                agent=agent,
                task=task,
                task_key=task_key,
                extra_inputs={"context": self.results["preparation"]},
            )
            print(f"[PHASE 2 - Task {i}/{len(analysis_order)}] ✓ Completed (or max retries reached)")

        self.results["analysis"] = "\n---\n".join(
            self.results.get(task_key, "") for task_key, _ in analysis_order
        )
        print("\n[PHASE 2] Analysis phase finished (with retries where needed)")

        print("\n[RATE LIMITING] Waiting 6 seconds before final report...")
        time.sleep(6)

        # ----- PHASE 3: REPORT GENERATION (MARKDOWN) -----
        print("\n" + "="*70)
        print("PHASE 3: REPORT GENERATION (MARKDOWN)")
        print("="*70)

        report_task = Task(
            description=(
                "Using the analysis context below, write a concise markdown report.\n\n"
                "=== PREPARATION PHASE ===\n"
                f"{self.results.get('preparation', 'N/A')}\n\n"
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

    # ------------------------------------------------------------------ #
    # Report generation helpers
    # ------------------------------------------------------------------ #
    def generate_markdown_report(self) -> str:
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
        return self.generate_markdown_report()


# ============================================================================
# PART 6: SAMPLE DATA + MAIN (UNCHANGED API)
# ============================================================================

def create_sample_dataset(path: str):
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
