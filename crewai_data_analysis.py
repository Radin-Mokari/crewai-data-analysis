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
        # Store output_dir so agents can use it for saving charts
        chart_output_dir = Path(self.output_dir).resolve()
        chart_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a wrapped savefig that always saves to the correct directory
        original_savefig = plt.savefig
        def wrapped_savefig(fname, *args, **kwargs):
            """Redirect all savefig calls to the chart output directory."""
            fname_path = Path(fname)
            # If it's just a filename (no directory), save to chart_output_dir
            if fname_path.parent == Path('.') or not fname_path.is_absolute():
                fname = str(chart_output_dir / fname_path.name)
            return original_savefig(fname, *args, **kwargs)
        
        # Replace plt.savefig with our wrapped version
        plt.savefig = wrapped_savefig
        
        base_globals = {
            "__name__": "__session__",
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "matplotlib": matplotlib,
            "CHART_OUTPUT_DIR": str(chart_output_dir),
            "Path": Path,
        }
        self.session_globals = base_globals

    def init_session(self, dataset_path: str):
        """
        Called once from workflow before any LLM tool calls.

        - Loads CSV into df_raw
        - Sets df_clean and df_features to None initially
        - Injects column metadata for dynamic column awareness (Core Mode)
        """
        code = f"""
import pandas as _pd
import numpy as _np

dataset_path = r\"\"\"{dataset_path}\"\"\"
df_raw = _pd.read_csv(dataset_path)
df_clean = None
df_features = None
validation_report = {{}}

# === CORE MODE METADATA ===
# These variables provide dynamic column awareness for all agents
DATASET_COLUMNS = list(df_raw.columns)
NUMERIC_COLUMNS = df_raw.select_dtypes(include=[_np.number]).columns.tolist()
CATEGORICAL_COLUMNS = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
BOOLEAN_COLUMNS = df_raw.select_dtypes(include=['bool']).columns.tolist()
DATASET_SHAPE = df_raw.shape
DATASET_PATH = dataset_path

# Print metadata summary for agents to reference
print("=== CORE MODE: Dataset Metadata ===")
print(f"Shape: {{DATASET_SHAPE[0]}} rows x {{DATASET_SHAPE[1]}} columns")
print(f"All columns: {{DATASET_COLUMNS}}")
print(f"Numeric columns ({{len(NUMERIC_COLUMNS)}}): {{NUMERIC_COLUMNS}}")
print(f"Categorical columns ({{len(CATEGORICAL_COLUMNS)}}): {{CATEGORICAL_COLUMNS}}")
print("===================================")
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


def _make_summarizer_llm():
    """Create a free DeepSeek R1 LLM for context summarization via OpenRouter."""
    from crewai import LLM

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None  # Fallback to rule-based compression

    return LLM(
        model="openrouter/deepseek/deepseek-r1:free",  # openrouter/ prefix for CrewAI routing
        api_key=api_key,
        config={"max_output_tokens": 500}
    )


# ============================================================================
# PART 3: AGENT DEFINITIONS (STATE-AWARE)
# ============================================================================

def create_agents(executor_tool: PythonSessionTool) -> Dict[str, Agent]:
    llm_short = _make_gemini_llm(max_output_tokens=320, thinking_budget=0)
    llm_medium = _make_gemini_llm(max_output_tokens=640, thinking_budget=256)
    llm_long = _make_gemini_llm(max_output_tokens=1200, thinking_budget=512)

    # === CORE MODE INSTRUCTION (shared by all data prep agents) ===
    core_mode_instruction = (
        "You operate in CORE MODE within a persistent Python kernel. "
        "CRITICAL RULES: "
        "1) NEVER call pd.read_csv() - data is pre-loaded in df_raw. "
        "2) Use DATASET_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS for column names. "
        "3) Reference existing variables: df_raw, df_clean, df_features, validation_report. "
        "4) Output concise code, no conversational text."
    )

    agents = {
        "library_import": Agent(
            role="Python Environment Setup Specialist",
            goal="Verify the Python environment and confirm df_raw and metadata variables are accessible.",
            backstory=(
                f"{core_mode_instruction} "
                "Your job: Verify that df_raw exists and print DATASET_COLUMNS to confirm metadata is loaded. "
                "Do NOT reload any data. Just confirm the environment is ready."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_loading": Agent(
            role="Data Structure Summariser",
            goal="Summarise df_raw structure using pre-loaded metadata variables.",
            backstory=(
                f"{core_mode_instruction} "
                "Your job: Print df_raw.shape, use NUMERIC_COLUMNS and CATEGORICAL_COLUMNS to describe the schema. "
                "Show df_raw.head(3) and df_raw.dtypes. Do NOT call read_csv."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_inspection": Agent(
            role="Data Inspection Analyst",
            goal="Inspect df_raw for quality issues using dynamic column detection.",
            backstory=(
                f"{core_mode_instruction} "
                "Your job: Use NUMERIC_COLUMNS and CATEGORICAL_COLUMNS to inspect data quality. "
                "Check missing values with df_raw[DATASET_COLUMNS].isnull().sum(). "
                "Check duplicates with df_raw.duplicated().sum(). Output bullet-point diagnostics."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_validation": Agent(
            role="Data Validation Specialist",
            goal="Run validation rules on df_raw using dynamic columns and store in validation_report.",
            backstory=(
                f"{core_mode_instruction} "
                "Your job: Validate df_raw dynamically - iterate over NUMERIC_COLUMNS for range checks, "
                "CATEGORICAL_COLUMNS for cardinality checks. Store results in validation_report dict. "
                "Do NOT hardcode column names - use the metadata variables."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_cleaning": Agent(
            role="Data Cleaning Specialist",
            goal="Create df_clean from df_raw using validation_report findings.",
            backstory=(
                f"{core_mode_instruction} "
                "Your job: Create df_clean = df_raw.copy(), then clean based on validation_report. "
                "Use NUMERIC_COLUMNS for numeric imputation, CATEGORICAL_COLUMNS for categorical handling. "
                "Print each cleaning step. INSPECTOR MODE: If code fails, read traceback, fix, and retry."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "data_transformation": Agent(
            role="Feature Engineering Expert",
            goal="Create df_features from df_clean with derived features for ML.",
            backstory=(
                f"{core_mode_instruction} "
                "Your job: Create df_features = df_clean.copy(). Engineer features using NUMERIC_COLUMNS "
                "(scaling, interactions) and CATEGORICAL_COLUMNS (encoding). "
                "Update NUMERIC_COLUMNS and CATEGORICAL_COLUMNS after transformations. "
                "Print summary of new features created."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        # === ANALYSIS AGENTS: Codified Prompting + Inspector Pattern ===
        "eda_analysis": Agent(
            role="Exploratory Data Analysis Specialist",
            goal="Perform EDA using CODIFIED PROMPTING - output pseudocode plan first, then execute.",
            backstory=(
                "You use CODIFIED PROMPTING: Output your analysis as structured pseudocode BEFORE executing. "
                "PLAN FORMAT:\n"
                "```\n"
                "def perform_eda(df):\n"
                "    # Step 1: Select best dataframe\n"
                "    # Step 2: Compute stats for NUMERIC_COLUMNS\n"
                "    # Step 3: Compute correlations\n"
                "    # Step 4: Identify patterns\n"
                "```\n"
                "Then execute. Use df_features if not None, else df_clean, else df_raw. "
                "INSPECTOR MODE: If execution fails, read traceback, fix code, retry (max 3 attempts)."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        "visualizations": Agent(
            role="Data Visualization Specialist",
            goal="Generate charts using CODIFIED PROMPTING - plan visualizations first, then create.",
            backstory=(
                "You use CODIFIED PROMPTING: Output your visualization plan as pseudocode FIRST.\n"
                "PLAN FORMAT:\n"
                "```\n"
                "def create_visualizations(df):\n"
                "    # Chart 1: Distribution of first numeric column\n"
                "    # Chart 2: Correlation heatmap of NUMERIC_COLUMNS\n"
                "    # Chart 3: Box plot for outlier detection\n"
                "```\n"
                "Then execute. Use NUMERIC_COLUMNS for dynamic column selection. "
                "DO NOT call plt.savefig() - tool saves automatically. "
                "INSPECTOR MODE: If plt errors occur, check column exists, fix, retry."
            ),
            llm=llm_short,
            tools=[executor_tool],
            verbose=True,
        ),
        "statistical_tests": Agent(
            role="Statistical Analysis Expert",
            goal="Run statistical tests using CODIFIED PROMPTING with dynamic column selection.",
            backstory=(
                "You use CODIFIED PROMPTING: Output test plan as pseudocode FIRST.\n"
                "PLAN FORMAT:\n"
                "```\n"
                "def run_statistical_tests(df):\n"
                "    # Test 1: Normality test on first NUMERIC_COLUMN\n"
                "    # Test 2: Correlation test between two NUMERIC_COLUMNS\n"
                "    # Test 3: Group comparison if CATEGORICAL_COLUMNS exist\n"
                "```\n"
                "CRITICAL: Before using ANY column, verify it exists: `if col in df.columns`.\n"
                "Use NUMERIC_COLUMNS[0], NUMERIC_COLUMNS[1] etc. - NEVER hardcode column names.\n"
                "INSPECTOR MODE: If KeyError occurs, print df.columns, select valid column, retry."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
        ),
        # === REPORT AGENT: Token Stratification ===
        "report_generator": Agent(
            role="Technical Report Writer",
            goal="Produce a markdown report using TOKEN STRATIFICATION - extract only high-value insights.",
            backstory=(
                "You use TOKEN STRATIFICATION to filter context efficiently.\n"
                "IGNORE (low-value tokens):\n"
                "- Code syntax and implementation details\n"
                "- Execution logs and tracebacks\n"
                "- Raw dataframe outputs and intermediate steps\n"
                "- Verbose debugging information\n\n"
                "EXTRACT ONLY (high-value tokens):\n"
                "- Dataset shape and column summary\n"
                "- Data quality issues found and how they were resolved\n"
                "- Statistical metrics: correlations, p-values, test results\n"
                "- Visualization file paths (charts created)\n"
                "- Key patterns, anomalies, and insights\n"
                "- Final recommendations for ML modeling\n\n"
                "Generate a CONCISE markdown report with these sections:\n"
                "# Executive Summary, ## Data Overview, ## Data Quality, "
                "## Key Findings, ## Statistical Results, ## Recommendations"
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
    # === CORE MODE TASK PREFIX ===
    core_mode_prefix = (
        "CORE MODE ACTIVE: Use pre-loaded variables (df_raw, DATASET_COLUMNS, "
        "NUMERIC_COLUMNS, CATEGORICAL_COLUMNS). NEVER call pd.read_csv().\n\n"
    )
    
    tasks = {
        "library_import": Task(
            description=(
                f"{core_mode_prefix}"
                "TASK: Verify environment is ready.\n"
                "CODE TO EXECUTE:\n"
                "```\n"
                "print('Environment Check:')\n"
                "print(f'df_raw loaded: {\"df_raw\" in dir()}')\n"
                "print(f'Shape: {DATASET_SHAPE}')\n"
                "print(f'Columns available: {len(DATASET_COLUMNS)}')\n"
                "print(f'Numeric: {NUMERIC_COLUMNS}')\n"
                "print(f'Categorical: {CATEGORICAL_COLUMNS}')\n"
                "```\n"
                "Output ONLY this verification code. No conversation."
            ),
            expected_output="Environment verification output showing df_raw and metadata are loaded.",
            agent=agents["library_import"],
            async_execution=False,
        ),
        "data_loading": Task(
            description=(
                f"{core_mode_prefix}"
                "TASK: Summarize df_raw structure using metadata variables.\n"
                "REQUIRED OUTPUT (as Python code):\n"
                "1. print(f'Shape: {df_raw.shape}')\n"
                "2. print(f'Columns: {DATASET_COLUMNS}')\n"
                "3. print(df_raw.dtypes)\n"
                "4. print(df_raw[DATASET_COLUMNS].isnull().sum())\n"
                "5. print(df_raw.head(3))\n"
                "Do NOT reload data. Use existing variables only."
            ),
            expected_output="Structured summary of df_raw using metadata variables.",
            agent=agents["data_loading"],
            async_execution=False,
        ),
        "data_inspection": Task(
            description=(
                f"{core_mode_prefix}"
                "TASK: Inspect data quality using dynamic column references.\n"
                "REQUIRED CHECKS:\n"
                "1. For col in NUMERIC_COLUMNS: print min, max, null count\n"
                "2. For col in CATEGORICAL_COLUMNS: print unique count, top values\n"
                "3. print(f'Duplicate rows: {df_raw.duplicated().sum()}')\n"
                "4. Print 2-3 bullet points summarizing quality issues found.\n"
                "Use NUMERIC_COLUMNS and CATEGORICAL_COLUMNS - do NOT hardcode column names."
            ),
            expected_output="Quality inspection report using dynamic column detection.",
            agent=agents["data_inspection"],
            async_execution=False,
        ),
        "data_validation": Task(
            description=(
                f"{core_mode_prefix}"
                "TASK: Validate df_raw and populate validation_report dict.\n"
                "VALIDATION RULES (iterate dynamically):\n"
                "```\n"
                "validation_report = {}\n"
                "validation_report['missing'] = df_raw[DATASET_COLUMNS].isnull().sum().to_dict()\n"
                "validation_report['duplicates'] = int(df_raw.duplicated().sum())\n"
                "for col in NUMERIC_COLUMNS:\n"
                "    validation_report[f'{col}_range'] = (df_raw[col].min(), df_raw[col].max())\n"
                "for col in CATEGORICAL_COLUMNS:\n"
                "    validation_report[f'{col}_unique'] = df_raw[col].nunique()\n"
                "```\n"
                "Print summary of validation_report. Do NOT create new DataFrames."
            ),
            expected_output="validation_report dict populated with dynamic validation results.",
            agent=agents["data_validation"],
            async_execution=False,
        ),
        "data_cleaning": Task(
            description=(
                f"{core_mode_prefix}"
                "TASK: Create df_clean from df_raw based on validation_report.\n"
                "CLEANING STEPS (use dynamic columns):\n"
                "1. df_clean = df_raw.copy()\n"
                "2. For col in NUMERIC_COLUMNS: impute missing with median\n"
                "3. For col in CATEGORICAL_COLUMNS: impute missing with mode or 'Unknown'\n"
                "4. Drop duplicate rows: df_clean.drop_duplicates(inplace=True)\n"
                "5. Print each operation performed\n"
                "6. Print final df_clean.shape\n\n"
                "INSPECTOR MODE: If code fails, read the traceback, identify the error, "
                "fix it, and retry. Do NOT ask for help."
            ),
            expected_output="df_clean created with cleaning steps documented.",
            agent=agents["data_cleaning"],
            async_execution=False,
        ),
        "data_transformation": Task(
            description=(
                f"{core_mode_prefix}"
                "TASK: Create df_features from df_clean for ML readiness.\n"
                "TRANSFORMATION STEPS:\n"
                "1. df_features = df_clean.copy()\n"
                "2. For NUMERIC_COLUMNS: Apply StandardScaler or MinMaxScaler\n"
                "3. For CATEGORICAL_COLUMNS: Apply LabelEncoder or pd.get_dummies()\n"
                "4. Create 1-2 derived features if meaningful (e.g., ratios, interactions)\n"
                "5. Update: NUMERIC_COLUMNS = df_features.select_dtypes(include=[np.number]).columns.tolist()\n"
                "6. Print transformations applied and df_features.shape\n\n"
                "INSPECTOR MODE: If encoding fails (e.g., unseen categories), catch error and use fallback."
            ),
            expected_output="df_features created with ML-ready transformations.",
            agent=agents["data_transformation"],
            async_execution=False,
        ),
        # === ANALYSIS TASKS: Codified Prompting + Inspector Pattern ===
        "eda_analysis": Task(
            description=(
                "CODIFIED PROMPTING: Output your plan as pseudocode FIRST, then execute.\n\n"
                "PSEUDOCODE PLAN:\n"
                "```\n"
                "def perform_eda():\n"
                "    # Step 1: Select dataframe (df_features if not None, else df_clean, else df_raw)\n"
                "    df = df_features if df_features is not None else (df_clean if df_clean is not None else df_raw)\n"
                "    # Step 2: Get current numeric columns\n"
                "    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
                "    # Step 3: Descriptive stats for numeric columns\n"
                "    print(df[num_cols].describe())\n"
                "    # Step 4: Correlation matrix\n"
                "    if len(num_cols) >= 2: print(df[num_cols].corr())\n"
                "    # Step 5: Print top 3 correlations and patterns\n"
                "```\n"
                "EXECUTE the plan. Output structured findings, not conversational text.\n\n"
                "INSPECTOR MODE: If execution fails, read traceback, fix, retry (max 3 attempts)."
            ),
            expected_output="EDA findings as structured output from codified plan execution.",
            agent=agents["eda_analysis"],
            async_execution=False,
        ),
        "visualizations": Task(
            description=(
                "CODIFIED PROMPTING: Output your visualization plan FIRST, then execute.\n\n"
                "PSEUDOCODE PLAN:\n"
                "```\n"
                "def create_visualizations():\n"
                "    df = df_features if df_features is not None else df_clean\n"
                "    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
                "    \n"
                "    # Chart 1: Distribution of first numeric column\n"
                "    if len(num_cols) >= 1:\n"
                "        plt.figure(figsize=(8,5))\n"
                "        sns.histplot(df[num_cols[0]], kde=True)\n"
                "        plt.title(f'Distribution of {num_cols[0]}')\n"
                "        plt.tight_layout()\n"
                "    \n"
                "    # Chart 2: Correlation heatmap\n"
                "    if len(num_cols) >= 2:\n"
                "        plt.figure(figsize=(10,8))\n"
                "        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')\n"
                "        plt.title('Correlation Heatmap')\n"
                "        plt.tight_layout()\n"
                "    \n"
                "    # Chart 3: Box plot for outliers\n"
                "    if len(num_cols) >= 1:\n"
                "        plt.figure(figsize=(10,6))\n"
                "        df[num_cols[:min(5, len(num_cols))]].boxplot()\n"
                "        plt.title('Box Plot - Outlier Detection')\n"
                "        plt.tight_layout()\n"
                "```\n"
                "EXECUTE the plan. DO NOT call plt.savefig() - tool saves automatically.\n\n"
                "INSPECTOR MODE: If column not found, use num_cols list, fix, retry."
            ),
            expected_output="Charts created using dynamic column selection from codified plan.",
            agent=agents["visualizations"],
            async_execution=False,
        ),
        "statistical_tests": Task(
            description=(
                "CODIFIED PROMPTING: Output your test plan FIRST, then execute.\n\n"
                "PSEUDOCODE PLAN:\n"
                "```\n"
                "def run_statistical_tests():\n"
                "    from scipy import stats\n"
                "    df = df_features if df_features is not None else (df_clean if df_clean is not None else df_raw)\n"
                "    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
                "    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()\n"
                "    \n"
                "    # Test 1: Normality test on first numeric column\n"
                "    if len(num_cols) >= 1:\n"
                "        col = num_cols[0]\n"
                "        stat, p = stats.shapiro(df[col].dropna().head(5000))\n"
                "        print(f'Normality Test ({col}): p-value = {p:.4f}')\n"
                "    \n"
                "    # Test 2: Correlation test between first two numeric columns\n"
                "    if len(num_cols) >= 2:\n"
                "        col1, col2 = num_cols[0], num_cols[1]\n"
                "        r, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())\n"
                "        print(f'Correlation Test ({col1} vs {col2}): r={r:.4f}, p-value={p:.4f}')\n"
                "    \n"
                "    # Test 3: Group comparison if categorical column exists\n"
                "    if len(cat_cols) >= 1 and len(num_cols) >= 1:\n"
                "        cat_col, num_col = cat_cols[0], num_cols[0]\n"
                "        groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]\n"
                "        if len(groups) >= 2:\n"
                "            stat, p = stats.kruskal(*groups[:5])  # Limit to 5 groups\n"
                "            print(f'Group Comparison ({num_col} by {cat_col}): p-value = {p:.4f}')\n"
                "```\n"
                "EXECUTE the plan. Use num_cols and cat_cols - NEVER hardcode column names.\n\n"
                "INSPECTOR MODE: If KeyError, print available columns, select valid one, retry."
            ),
            expected_output="Statistical test results using dynamic column selection.",
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
        
        # Generate unique run ID based on timestamp for this execution
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / f"run_{self.run_id}"
        self.run_output_dir.mkdir(exist_ok=True)

        # Shared, long-lived Python session tool - charts go to run-specific directory
        self.executor = PythonSessionTool(output_dir=str(self.run_output_dir / "charts"))
        # Load the dataset once into df_raw
        self.executor.init_session(self.dataset_path)

        self.agents = create_agents(self.executor)
        self.tasks = create_tasks(self.agents)

        self.results: Dict[str, Any] = {}
        self.charts: List[Path] = []
        self.report_path: Optional[Path] = None

        # Initialize DeepSeek R1 summarizer for context compression
        self.summarizer_llm = _make_summarizer_llm()
        if self.summarizer_llm:
            print("[OK] DeepSeek R1 summarizer enabled (OpenRouter)")
        else:
            print("[INFO] No OPENROUTER_API_KEY - using rule-based compression")

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
    # Context Compression (DeepSeek R1 or rule-based fallback)
    # ------------------------------------------------------------------ #
    def _rule_based_compress(self, text: str, max_chars: int = 2000) -> str:
        """Fallback compression using regex (no LLM call)."""
        import re
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '[CODE BLOCK REMOVED]', text)
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[TRUNCATED]"
        return text

    def _summarize_context(self, raw_text: str, phase_name: str) -> str:
        """Compress context using DeepSeek R1 FREE or fallback to rule-based."""
        if not self.summarizer_llm:
            return self._rule_based_compress(raw_text)

        prompt = f"""Summarize this {phase_name} output in 5-7 bullet points.
EXTRACT ONLY: dataset shape, column names, metrics, errors, key findings.
IGNORE: code blocks, tracebacks, verbose logs.

OUTPUT:
{raw_text[:8000]}"""

        try:
            print(f"[SUMMARIZER] Calling DeepSeek R1 for {phase_name} compression...")
            response = self.summarizer_llm.call(prompt)
            return str(response)
        except Exception as e:
            print(f"[SUMMARIZER] DeepSeek R1 failed, using fallback: {e}")
            return self._rule_based_compress(raw_text)

    # ------------------------------------------------------------------ #
    # Main pipeline
    # ------------------------------------------------------------------ #
    def run_sequential_pipeline(self) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print("STARTING SEQUENTIAL DATA ANALYSIS PIPELINE (STATEFUL)")
        print(f"Dataset: {self.dataset_path}")
        print(f"Run ID: {self.run_id}")
        print(f"Output Directory: {self.run_output_dir}")
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
            print(f"[PHASE 1 - Task {i}/{len(prep_order)}] [OK] Completed (or max retries reached)")

        # Compress preparation results before Phase 2
        raw_prep = "\n---\n".join(
            self.results.get(task_key, "") for task_key, _ in prep_order
        )
        self.results["preparation"] = self._summarize_context(raw_prep, "Data Preparation")
        print(f"[SUMMARIZER] Preparation context compressed: {len(raw_prep)} -> {len(self.results['preparation'])} chars")
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
            print(f"[PHASE 2 - Task {i}/{len(analysis_order)}] [OK] Completed (or max retries reached)")

        # Compress analysis results before Phase 3
        raw_analysis = "\n---\n".join(
            self.results.get(task_key, "") for task_key, _ in analysis_order
        )
        self.results["analysis"] = self._summarize_context(raw_analysis, "Analysis")
        print(f"[SUMMARIZER] Analysis context compressed: {len(raw_analysis)} -> {len(self.results['analysis'])} chars")
        print("\n[PHASE 2] Analysis phase finished (with retries where needed)")

        print("\n[RATE LIMITING] Waiting 6 seconds before final report...")
        time.sleep(6)

        # ----- PHASE 3: REPORT GENERATION (MARKDOWN) -----
        print("\n" + "="*70)
        print("PHASE 3: REPORT GENERATION (MARKDOWN)")
        print("="*70)

        report_task = Task(
            description=(
                "TOKEN STRATIFICATION: Generate a markdown report by extracting ONLY high-value insights.\n\n"
                "=== CONTEXT (Apply Token Stratification - ignore code, extract insights) ===\n\n"
                "PREPARATION PHASE RESULTS:\n"
                f"{self.results.get('preparation', 'N/A')}\n\n"
                "ANALYSIS PHASE RESULTS:\n"
                f"{self.results.get('analysis', 'N/A')}\n\n"
                "=== TOKEN STRATIFICATION RULES ===\n"
                "IGNORE: Code blocks, tracebacks, raw dataframe outputs, verbose logs\n"
                "EXTRACT: Dataset shape, column names, quality issues, correlations, p-values, chart paths, patterns\n\n"
                "=== REQUIRED OUTPUT FORMAT ===\n"
                "# Executive Summary\n"
                "(2-3 sentences: dataset size, main findings, ML readiness)\n\n"
                "## Data Overview\n"
                "(Shape, column types, target variable if identifiable)\n\n"
                "## Data Quality & Cleaning\n"
                "(Issues found, cleaning steps applied)\n\n"
                "## Key Findings\n"
                "(Top correlations, patterns, anomalies - bullet points)\n\n"
                "## Statistical Results\n"
                "(Test names, variables, p-values, interpretations)\n\n"
                "## Recommendations\n"
                "(Next steps for ML modeling)\n\n"
                "Max 800 words. No code blocks. Bullet points preferred."
            ),
            expected_output="Concise markdown report using Token Stratification (<= 800 words).",
            agent=self.agents["report_generator"],
        )

        try:
            print("\n[PHASE 3] Starting report generation...")
            print(f"[TIME] {datetime.now().strftime('%H:%M:%S')} - Starting report task...")
            report_result = report_task.execute_sync(agent=self.agents["report_generator"])
            self.results["report"] = str(report_result)
            print("\n[PHASE 3] [OK] Report generation completed")
        except Exception as e:
            print(f"\n[PHASE 3] [ERROR] Error in report generation: {e}")
            # Generate a fallback report from the collected results
            self.results["report"] = self._generate_fallback_report()

        # Collect charts from run-specific directory and save report immediately
        charts_dir = self.run_output_dir / "charts"
        if charts_dir.exists():
            self.charts = list(charts_dir.glob("*.png"))
        else:
            self.charts = []
        
        # Save the report now to ensure it's written even if something fails later
        self._save_report_to_file()
        
        return self.results
    
    def _generate_fallback_report(self) -> str:
        """Generate a fallback report from collected results if LLM fails."""
        report_lines = [
            "# Data Analysis Report",
            "",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Executive Summary",
            "",
            "This report was auto-generated from the analysis pipeline results.",
            "",
            "## Preparation Phase Results",
            "",
            "```",
            self.results.get("preparation", "No preparation results available.")[:2000],
            "```",
            "",
            "## Analysis Phase Results", 
            "",
            "```",
            self.results.get("analysis", "No analysis results available.")[:2000],
            "```",
            "",
            "## Charts Generated",
            "",
        ]
        
        charts_dir = self.run_output_dir / "charts"
        if charts_dir.exists():
            for chart in charts_dir.glob("*.png"):
                report_lines.append(f"- {chart.name}")
        else:
            report_lines.append("No charts were generated.")
        
        return "\n".join(report_lines)
    
    def _save_report_to_file(self):
        """Save the report to file immediately in the run-specific directory."""
        report_md = self.results.get("report", "")
        if not report_md or report_md.startswith("Error:"):
            report_md = self._generate_fallback_report()
        
        # Save to run-specific directory with unique filename
        self.report_path = self.run_output_dir / f"analysis_report_{self.run_id}.md"
        try:
            self.report_path.write_text(report_md, encoding="utf-8")
            print(f"[REPORT] [OK] Report saved to: {self.report_path}")
        except Exception as e:
            print(f"[REPORT] [ERROR] Failed to save report: {e}")

    # ------------------------------------------------------------------ #
    # Report generation helpers
    # ------------------------------------------------------------------ #
    def generate_markdown_report(self) -> str:
        """Generate and save the markdown report. Report may already be saved."""
        # Use run-specific report path
        report_path = self.report_path or (self.run_output_dir / f"analysis_report_{self.run_id}.md")
        
        # Check if report was already saved during pipeline
        if report_path.exists():
            print(f"\n{'='*70}")
            print("[OK] MARKDOWN REPORT ALREADY GENERATED")
            print(f"Location: {report_path}")
            print(f"Run directory: {self.run_output_dir}")
            print(f"{'='*70}\n")
            return str(report_path)
        
        # If not, save it now
        report_md = self.results.get("report", "")
        if not report_md or report_md.startswith("Error:"):
            report_md = self._generate_fallback_report()

        try:
            report_path.write_text(report_md, encoding="utf-8")
            self.report_path = report_path
            print(f"\n{'='*70}")
            print("[OK] MARKDOWN REPORT GENERATED")
            print(f"Location: {report_path}")
            print(f"Run directory: {self.run_output_dir}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[ERROR] FAILED TO SAVE REPORT: {e}")
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
    print(f"[OK] Sample dataset created: {path}")
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
