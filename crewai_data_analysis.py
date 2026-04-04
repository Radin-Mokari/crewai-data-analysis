# ============================================================================
# COMPLETE CREWAI DATA ANALYSIS WORKFLOW (STATEFUL, TOKEN-OPTIMIZED)
# ============================================================================

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, Tuple
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import Field, BaseModel

import google.generativeai as genai

from session_state_store import save_kernel_snapshot, try_load_kernel_snapshot


@dataclass
class InteractiveSegmentResult:
    """One user message's supervisor inner loop (until CHAT, DONE, or guardrail)."""

    outcome: Literal["await_user", "session_exit_manager_cap"]


@dataclass
class SupervisorLoopState:
    """Mutable counters for one dynamic supervisor run (batch or interactive)."""

    last_agent: Optional[str] = None
    repeat_streak: int = 0
    specialist_count: int = 0
    manager_invocations: int = 0


SupervisorSingleTurnOutcome = Literal[
    "continue",
    "done",
    "break_interactive",
    "exit_manager_cap",
    "exit_specialist_cap",
    "exit_repeat",
]


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
        """Initialize shared session globals."""
        chart_output_dir = Path(self.output_dir).resolve()
        chart_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Intercept savefig to force output to our charts directory
        original_savefig = plt.savefig
        def wrapped_savefig(fname, *args, **kwargs):
            fname_path = Path(fname)
            if fname_path.parent == Path('.') or not fname_path.is_absolute():
                fname = str(chart_output_dir / fname_path.name)
            return original_savefig(fname, *args, **kwargs)
        
        plt.savefig = wrapped_savefig
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['font.size'] = 11
        
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
        """Load dataset and inject metadata for Core Mode."""
        code = f"""
import pandas as _pd
import numpy as _np

dataset_path = r\"\"\"{dataset_path}\"\"\"
df_raw = _pd.read_csv(dataset_path)
df_clean = None
df_features = None
validation_report = []  # list of findings/messages; agents use .append()

# === CORE MODE METADATA ===
# These variables provide dynamic column awareness for all agents
DATASET_COLUMNS = list(df_raw.columns)
NUMERIC_COLUMNS = df_raw.select_dtypes(include=[_np.number]).columns.tolist()
CATEGORICAL_COLUMNS = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
BOOLEAN_COLUMNS = df_raw.select_dtypes(include=['bool']).columns.tolist()
DATASET_SHAPE = df_raw.shape
DATASET_PATH = dataset_path
TIME_INDEX_OK = False

# === PRESERVE ORIGINAL COLUMNS (before any transformations) ===
# These are used by visualization agent for interpretable charts
ORIGINAL_NUMERIC_COLUMNS = NUMERIC_COLUMNS.copy()
ORIGINAL_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS.copy()

# Print metadata summary for agents to reference
print("=== CORE MODE: Dataset Metadata ===")
print(f"Shape: {{DATASET_SHAPE[0]}} rows x {{DATASET_SHAPE[1]}} columns")
print(f"All columns: {{DATASET_COLUMNS}}")
print(f"Numeric columns ({{len(NUMERIC_COLUMNS)}}): {{NUMERIC_COLUMNS}}")
print(f"Categorical columns ({{len(CATEGORICAL_COLUMNS)}}): {{CATEGORICAL_COLUMNS}}")
print("===================================")

# === STATE VALIDATION HELPER ===
def validate_core_state():
    '''Call at start of each task to verify state is ready.'''
    g = globals()
    required = ['df_raw', 'NUMERIC_COLUMNS', 'CATEGORICAL_COLUMNS', 'DATASET_COLUMNS']
    missing = [v for v in required if v not in g or g.get(v) is None]
    if missing:
        raise RuntimeError(f'Missing required state: {{missing}}')
    print('State OK: All required variables present')
    print(f'  - NUMERIC_COLUMNS: {{len(NUMERIC_COLUMNS)}} columns')
    print(f'  - CATEGORICAL_COLUMNS: {{len(CATEGORICAL_COLUMNS)}} columns')
    return True
"""
        exec(code, self.session_globals)

    def validate_state(self) -> Dict[str, bool]:
        """Check presence of key objects."""
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
            "time_index_ok": bool(g.get("TIME_INDEX_OK")),
        }
        return flags


    def _run(self, code: str) -> str:
        """Execute Python code and return JSON with stdout, charts, state_flags."""
        result = {
            "success": False,
            "stdout": "",
            "error": None,
            "charts": [],
            "state_flags": {},
        }

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

        charts = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            chart_path = Path(self.output_dir) / f"chart_{int(time.time() * 1000)}_{fig_num}.png"
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            charts.append(str(chart_path))
            plt.close(fig)
        result["charts"] = charts

        result["state_flags"] = self.validate_state()

        # Fix Rich stdout corruption after redirect
        import sys
        if hasattr(sys, '__stdout__') and sys.__stdout__ is not None:
            sys.stdout = sys.__stdout__

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

    core_mode_instruction = (
        "You operate in CORE MODE within a persistent Python kernel. "
        "CRITICAL RULES: "
        "1) NEVER call pd.read_csv() - data is pre-loaded in df_raw. "
        "2) Use DATASET_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS for column names. "
        "3) Reference existing variables: df_raw, df_clean, df_features, validation_report. "
        "4) Output concise code, no conversational text. "
        "5) FINAL ANSWER FORMAT: Return a 2-3 sentence summary of what was done, NOT the full code."
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
            goal="Analyze data characteristics and generate the most insightful visualizations.",
            backstory=(
                "You are an INTELLIGENT visualization specialist who analyzes data BEFORE choosing charts.\n"
                "STEP 1 - ANALYZE DATA CHARACTERISTICS:\n"
                "- Compute skewness: highly skewed (|skew|>1) needs log-scale or box plot\n"
                "- Check cardinality: nunique<=10 use bar chart, nunique>20 use histogram\n"
                "- Find top correlations: |corr|>0.5 deserves scatter plot\n"
                "- Detect binary columns (nunique==2): use as grouping/hue variable\n"
                "STEP 2 - SELECT CHARTS based on insights, not fixed templates.\n"
                "CRITICAL: Use df_clean (not df_features) for interpretable values.\n"
                "Use ORIGINAL_NUMERIC_COLUMNS for original column names.\n"
                "DO NOT call plt.savefig() - tool saves automatically."
            ),
            llm=llm_medium,
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
        "report_generator": Agent(
            role="Technical Report Writer",
            goal="Produce a concise markdown report summarizing data analysis results.",
            backstory=(
                "You are a technical writer who creates clear, professional data analysis reports. "
                "You extract key insights from analysis results and present them in a structured format. "
                "You focus on: dataset characteristics, data quality, statistical findings, and recommendations. "
                "You output ONLY the markdown report with no explanations or reasoning - just the content."
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
    core_mode_prefix = "CORE MODE: Use existing df_raw, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS.\n\n"
    
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
                "TASK: Create df_features from df_clean for ML readiness.\n\n"
                "STEP 0 (REQUIRED FIRST): Print state verification:\n"
                "```\n"
                "print(f'NUMERIC_COLUMNS: {NUMERIC_COLUMNS}')\n"
                "print(f'CATEGORICAL_COLUMNS: {CATEGORICAL_COLUMNS}')\n"
                "print(f'df_clean shape: {df_clean.shape}')\n"
                "```\n\n"
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
                "INTELLIGENT VISUALIZATION: Analyze data characteristics, then create insightful charts.\n\n"
                "STEP 1 - ANALYZE (execute this code first):\n"
                "```\n"
                "from scipy.stats import skew\n"
                "df = df_clean if df_clean is not None else df_raw\n"
                "num_cols = ORIGINAL_NUMERIC_COLUMNS\n"
                "cat_cols = ORIGINAL_CATEGORICAL_COLUMNS\n"
                "\n"
                "# Compute characteristics\n"
                "skewness = {col: skew(df[col].dropna()) for col in num_cols if col in df.columns}\n"
                "highly_skewed = [c for c,s in skewness.items() if abs(s) > 1]\n"
                "binary_cols = [c for c in df.columns if df[c].nunique() == 2]\n"
                "\n"
                "# Find top correlations\n"
                "if len(num_cols) >= 2:\n"
                "    corr = df[num_cols].corr().abs()\n"
                "    pairs = [(corr.loc[i,j],i,j) for i in num_cols for j in num_cols if i<j]\n"
                "    top_corr = sorted(pairs, reverse=True)[:3]\n"
                "    print(f'Top correlations: {top_corr}')\n"
                "print(f'Highly skewed columns: {highly_skewed}')\n"
                "print(f'Binary columns for grouping: {binary_cols}')\n"
                "```\n\n"
                "STEP 2 - CREATE CHARTS based on analysis:\n"
                "- For each highly_skewed col: box plot (better than histogram)\n"
                "- For non-skewed numeric: histogram with KDE\n"
                "- For top correlated pairs (|r|>0.5): scatter plot with trend line\n"
                "- If binary_cols exist: use as hue in scatter/violin plots\n"
                "- Correlation heatmap: limit to top 10 cols by variance\n"
                "- If cat_cols exist: grouped bar or violin plot\n\n"
                "IMPORTANT: Use descriptive titles with column names. Example:\n"
                "plt.title(f'Distribution of {col_name} (skewness: {skewness[col_name]:.2f})')\n\n"
                "DO NOT call plt.savefig(). Create 3-5 insightful charts total."
            ),
            expected_output="Insightful charts selected based on data characteristics analysis.",
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
# PART 4B: DYNAMIC SUPERVISOR (brief, specialists, manager JSON, chat persist)
# ============================================================================

DYNAMIC_CORE_MODE = (
    "You operate in CORE MODE within a persistent Python kernel. "
    "CRITICAL RULES: "
    "1) NEVER call pd.read_csv() - data is pre-loaded in df_raw. "
    "2) Use DATASET_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS for column names. "
    "3) Reference existing variables: df_raw, df_clean, df_features, validation_report. "
    "4) Output concise code, no conversational text. "
    "5) FINAL ANSWER FORMAT: Return a 2-3 sentence summary of what was done, NOT the full code."
)


class ManagerDecision(BaseModel):
    next_agent: Literal[
        "cleaning",
        "feature_engineering",
        "class_imbalance",
        "eda",
        "visualization",
        "statistics",
        "reporter",
        "CHAT",
        "DONE",
    ]
    instruction: str = ""
    rationale: str = ""
    reply_to_user: Optional[str] = None  # Gemini often returns null; treat as "" everywhere


def compute_dataset_brief(executor: PythonSessionTool, run_output_dir: Path) -> Tuple[str, Dict[str, Any]]:
    code = r"""
import pandas as _pd
lines = []
brief = {"is_time_series": False, "time_column_candidates": [], "shape": None, "columns": []}
df = df_raw
brief["shape"] = list(df.shape)
brief["columns"] = list(DATASET_COLUMNS)
for col in DATASET_COLUMNS:
    s = df[col]
    try:
        if _pd.api.types.is_datetime64_any_dtype(s):
            brief["time_column_candidates"].append({"column": col, "reason": "datetime64_dtype"})
            continue
    except Exception:
        pass
    if s.dtype == object or str(s.dtype) == "string":
        sample = s.head(min(500, len(s)))
        try:
            parsed = _pd.to_datetime(sample, errors="coerce")
            rate = float(parsed.notna().mean()) if len(sample) else 0.0
            if rate >= 0.75:
                brief["time_column_candidates"].append(
                    {"column": col, "reason": f"parseable_object_rate_{rate:.2f}"}
                )
        except Exception:
            pass
brief["is_time_series"] = len(brief["time_column_candidates"]) > 0
lines.append("=== DATASET BRIEF (deterministic) ===")
lines.append(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")
lines.append(f"is_time_series: {brief['is_time_series']}")
if brief["time_column_candidates"]:
    lines.append("time_column_candidates:")
    for tc in brief["time_column_candidates"][:12]:
        lines.append(f"  {tc}")
missing = df[DATASET_COLUMNS].isnull().sum().sort_values(ascending=False)
top_miss = missing[missing > 0].head(10)
if len(top_miss):
    lines.append("top_missing_counts:")
    for c, v in top_miss.items():
        lines.append(f"  {c}: {int(v)}")
lines.append(f"duplicate_rows: {int(df.duplicated().sum())}")
DATASET_BRIEF_DICT = brief
DATASET_BRIEF_TEXT = "\n".join(lines)
print(DATASET_BRIEF_TEXT)
"""
    executor._run(code)
    brief_text = str(executor.session_globals.get("DATASET_BRIEF_TEXT", ""))
    brief_dict = executor.session_globals.get("DATASET_BRIEF_DICT")
    if not isinstance(brief_dict, dict):
        brief_dict = {"is_time_series": False, "time_column_candidates": [], "shape": [], "columns": []}
    out_path = Path(run_output_dir) / "dataset_brief.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(brief_text, encoding="utf-8")
    return brief_text, brief_dict


def persist_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def manager_records_to_chat_block(records: List[Dict[str, Any]], max_items: int = 28) -> str:
    """Build prompt block; skips `chat_turn` rows (analytics-only duplicates of CHAT replies)."""
    parts: List[str] = []
    for rec in records[-max_items:]:
        if rec.get("kind") == "chat_turn":
            continue
        kind = rec.get("kind", "message")
        role = rec.get("role", "user")
        content = str(rec.get("content", ""))[:12000]
        ts = rec.get("ts", "")
        parts.append(f"[{ts}][{kind}|{role}]\n{content}")
    return "\n\n".join(parts)


def summarize_manager_chat_head(records: List[Dict[str, Any]]) -> str:
    """Compact older chat/analysis records for manager context (plan: hybrid memory)."""
    if not records:
        return ""
    api_key = os.getenv("GEMINI_API_KEY")
    lines: List[str] = []
    for r in records:
        if r.get("kind") == "chat_turn":
            continue
        lines.append(
            f"{r.get('kind', '')}|{r.get('role', '')}: {str(r.get('content', ''))[:3500]}"
        )
    blob = "\n".join(lines)
    if len(blob) > 120_000:
        blob = blob[:120_000] + "\n...(truncated for summarizer)"
    if not api_key:
        return "\n".join(f"- {line[:500]}" for line in lines[:30])
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(
            "Summarize these supervisor run records for continued routing. "
            "Bullets: user goals, manager JSON decisions, specialist outcomes, errors. Max ~400 words.\n\n"
            + blob
        )
        return (resp.text or "").strip() or "(empty summary)"
    except Exception as e:
        return f"(summarization failed: {e}); fallback:\n" + "\n".join(lines[:12])


def build_manager_chat_block_for_llm(
    records: List[Dict[str, Any]],
    *,
    run_output_dir: Optional[Path] = None,
    max_chars: int = 28000,
    keep_recent: int = 14,
) -> str:
    """Sliding window + rolling summary when the chat block exceeds a character budget."""
    if not records:
        return ""
    budget = int(os.getenv("MANAGER_CHAT_BUDGET_CHARS", str(max_chars)))
    tail_n = max(4, int(os.getenv("MANAGER_CHAT_KEEP_RECENT", str(keep_recent))))

    full_block = manager_records_to_chat_block(records, max_items=len(records))
    if len(full_block) <= budget:
        return full_block

    head = records[:-tail_n] if len(records) > tail_n else []
    tail = records[-tail_n:] if len(records) > tail_n else records
    summary = summarize_manager_chat_head(head) if head else ""
    if run_output_dir and summary:
        try:
            (Path(run_output_dir) / "conversation_summary.txt").write_text(summary, encoding="utf-8")
        except OSError:
            pass

    tail_block = manager_records_to_chat_block(tail, max_items=len(tail))
    combined = f"[EARLIER_CONTEXT_SUMMARY]\n{summary}\n\n---\n\n{tail_block}"
    if len(combined) > budget:
        tail_allow = max(2000, budget - len(summary) - 80)
        tail_block = tail_block[-tail_allow:]
        combined = f"[EARLIER_CONTEXT_SUMMARY]\n{summary}\n\n---\n\n{tail_block}"
    return combined


def parse_manager_decision(text: str) -> Optional[ManagerDecision]:
    raw = text.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group(0)
    try:
        return ManagerDecision.model_validate_json(raw)
    except Exception:
        return None


def invoke_manager_decision(
    *,
    chat_block: str,
    user_payload: str,
    system_instruction: str,
) -> ManagerDecision:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
    full_prompt = chat_block + "\n\n---\n\n" + user_payload
    last_snippet = ""
    for attempt in range(3):
        resp = model.generate_content(full_prompt)
        text = (resp.text or "").strip()
        decision = parse_manager_decision(text)
        if decision is not None:
            return decision
        last_snippet = text[:1800]
        full_prompt = (
            "Return ONLY JSON: "
            '{"next_agent":"...","instruction":"...","rationale":"...","reply_to_user":"..."} '
            "reply_to_user is required when next_agent is CHAT (user-visible answer). "
            "next_agent must be one of: cleaning, feature_engineering, class_imbalance, eda, visualization, "
            "statistics, reporter, CHAT, DONE. No markdown.\nInvalid output:\n"
            + last_snippet
        )
    raise ValueError(f"Manager JSON parse failed after retries. Last snippet: {last_snippet}")


def create_dynamic_specialist_agents(executor_tool: PythonSessionTool) -> Dict[str, Agent]:
    llm_medium = _make_gemini_llm(max_output_tokens=640, thinking_budget=256)
    llm_long = _make_gemini_llm(max_output_tokens=1200, thinking_budget=512)
    cm = DYNAMIC_CORE_MODE

    return {
        "cleaning": Agent(
            role="Data Cleaning & Preparation Specialist",
            goal="Verify environment, profile df_raw, validate, and produce df_clean using dynamic columns.",
            backstory=(
                f"{cm} You absorb former pipeline steps: env check, structure summary, quality inspection, "
                "validation_report population, and cleaning. For time-series data, set TIME_INDEX_OK = True only after "
                "df_clean is sorted by the primary time column. INSPECTOR MODE on errors."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
            allow_delegation=False,
        ),
        "feature_engineering": Agent(
            role="Feature Engineering Expert",
            goal="Create df_features from df_clean with ML-oriented transforms.",
            backstory=(
                f"{cm} For time-indexed data: lags/rolling use past values only (no future leakage). Do not shuffle rows. "
                "INSPECTOR MODE on errors."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
            allow_delegation=False,
        ),
        "class_imbalance": Agent(
            role="Class Imbalance Analyst",
            goal="Assess label distribution and imbalance using Core Mode columns.",
            backstory=(
                f"{cm} Detect plausible targets without hardcoding. Use value_counts and ratios. "
                "If labels are time-ordered, avoid suggesting random shuffles for balancing."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
            allow_delegation=False,
        ),
        "eda": Agent(
            role="Exploratory Data Analysis Specialist",
            goal="EDA with codified plan then execution.",
            backstory=(
                "CODIFIED PROMPTING: pseudocode plan first, then execute. "
                "Use df_features if not None else df_clean else df_raw. INSPECTOR MODE."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
            allow_delegation=False,
        ),
        "visualization": Agent(
            role="Data Visualization Specialist",
            goal="Insightful charts from df_clean for interpretability.",
            backstory=(
                "Analyze skew/cardinality/correlations before plotting. Prefer df_clean and ORIGINAL_* semantics. "
                "For time series use line/trend-style plots when appropriate. DO NOT call plt.savefig(); tool handles it."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
            allow_delegation=False,
        ),
        "statistics": Agent(
            role="Statistical Analysis Expert",
            goal="Statistical tests with codified prompting.",
            backstory=(
                "CODIFIED PLAN first, then execute. Verify columns exist. For time series, favor stationarity/autocorr checks "
                "when relevant."
            ),
            llm=llm_medium,
            tools=[executor_tool],
            verbose=True,
            allow_delegation=False,
        ),
        "reporter": Agent(
            role="Technical Report Writer",
            goal="Produce concise markdown using ONLY SESSION FACTS and step excerpts in the task—never generic templates.",
            backstory=(
                "Technical writer. OUTPUT ONLY markdown per task instructions—no preamble. Avoid code blocks unless asked. "
                "Never use [Number] or other bracket placeholders; every claim must trace to text in the task."
            ),
            llm=llm_long,
            tools=[],
            verbose=True,
            allow_delegation=False,
        ),
    }


def _ts_task_appendix(agent_id: str) -> str:
    common = (
        "\n\nTIME-SERIES PROTOCOL: preserve chronological order; never random row shuffling for train/test; "
        "no future leakage in derived features."
    )
    if agent_id == "cleaning":
        return common + (
            " Parse/normalize primary timestamp, sort rows, handle duplicate timestamps, report gaps. "
            "Set TIME_INDEX_OK = True in globals() only after df_clean is chronologically sorted."
        )
    if agent_id == "feature_engineering":
        return common + " Use shifts/rolling windows that only reference past values."
    if agent_id in ("eda", "visualization"):
        return common + " Prefer time-indexed visualizations (trend/seasonality)."
    if agent_id == "statistics":
        return common + " Prefer stationarity / autocorrelation checks when appropriate."
    if agent_id == "class_imbalance":
        return common + " Any balancing strategy must respect time order."
    return common


def build_specialist_task(
    *,
    agent_id: str,
    agents: Dict[str, Agent],
    user_prompt: str,
    manager_instruction: str,
    use_ts_appendix: bool,
    reporter_session_facts: Optional[str] = None,
    reporter_run_history: Optional[List[Dict[str, Any]]] = None,
) -> Task:
    core_mode_prefix = "CORE MODE: Use existing df_raw, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS. Do NOT read_csv.\n\n"
    ts_block = _ts_task_appendix(agent_id) if use_ts_appendix else ""
    header = (
        f"USER GOAL:\n{user_prompt}\n\n"
        f"MANAGER INSTRUCTION:\n{manager_instruction}\n"
        f"{ts_block}\n\n"
    )
    if agent_id == "cleaning":
        desc = (
            header
            + core_mode_prefix
            + "TASK: Full preparation — env check, schema/quality summary, populate validation_report, build df_clean; print shape."
        )
    elif agent_id == "feature_engineering":
        desc = (
            header
            + core_mode_prefix
            + "TASK: df_features from df_clean; encode/scale; update numeric/categoric column lists; summarize."
        )
    elif agent_id == "class_imbalance":
        desc = (
            header
            + core_mode_prefix
            + "TASK: Candidate targets via metadata; value_counts; imbalance metrics; recommendations."
        )
    elif agent_id == "eda":
        desc = header + "CODIFIED EDA: plan then execute on best df_*."
    elif agent_id == "visualization":
        desc = header + "SMART VIZ: 3–5 charts on df_clean using ORIGINAL_* where appropriate."
    elif agent_id == "statistics":
        desc = header + "CODIFIED STATISTICS: plan then tests with dynamic columns."
    elif agent_id == "reporter":
        desc = header + (manager_instruction or "Synthesize the final report.")
        blocks: List[str] = [desc, REPORTER_GROUNDING_RULES]
        if reporter_session_facts:
            blocks.append("SESSION FACTS (deterministic — prefer these numbers over memory):\n" + reporter_session_facts)
        if reporter_run_history is not None:
            digest = format_run_history_digest(
                reporter_run_history,
                last_n=50,
                max_instruction_chars=500,
                max_excerpt_chars=6000,
            )
            blocks.append("COMPLETED ANALYSIS STEPS (verbatim excerpts — your only source for process details):\n" + digest)
        desc = "\n\n".join(blocks)
    else:
        desc = header + core_mode_prefix + "Execute the manager instruction."

    return Task(
        description=desc,
        expected_output=f"Completed work for {agent_id} with concise summary.",
        agent=agents[agent_id],
        async_execution=False,
    )


def build_manager_system_instruction(brief_dict: Dict[str, Any]) -> str:
    ts = bool(brief_dict.get("is_time_series"))
    ts_rules = ""
    if ts:
        ts_rules = (
            " Dataset is flagged time-series: establish chronological df_clean first. Until the session marks "
            "TIME_INDEX_OK true, do not delegate feature_engineering. Forbid instructions that shuffle time order.\n"
        )
    return ts_rules + (
        "You route specialists for a shared Python data session. "
        "Output ONLY valid JSON (no markdown) with keys: next_agent, instruction, rationale, reply_to_user (optional string). "
        "next_agent must be one of: cleaning, feature_engineering, class_imbalance, eda, visualization, statistics, "
        "reporter, CHAT, DONE. "
        "Use CHAT when you should answer the user conversationally without running a specialist — set reply_to_user to the "
        "full user-visible answer (markdown/plain text); instruction may be empty. "
        "Use DONE when analysis is sufficient for a final report; you may set reply_to_user for a short closing note.\n"
        "Prefer delegating reporter only after df_clean exists and at least one analysis or visualization step has run, "
        "unless the user explicitly asks for an early summary.\n"
    )


def format_run_history_digest(
    entries: List[Dict[str, Any]],
    last_n: int = 5,
    *,
    max_instruction_chars: int = 320,
    max_excerpt_chars: int = 900,
) -> str:
    """Compact step list for prompts. Supervisor uses short excerpts; reporter/report use larger limits."""
    if not entries:
        return "(no steps yet)"
    lines: List[str] = []
    for e in entries[-last_n:]:
        instr = str(e.get("instruction", ""))[:max_instruction_chars]
        ex = str(e.get("output_excerpt", ""))[:max_excerpt_chars]
        lines.append(
            f"- step {e.get('step')}: {e.get('agent')} — instruction: {instr}\n" f"  excerpt: {ex}"
        )
    return "\n".join(lines)


def _report_text_has_placeholders(text: str) -> bool:
    """True if output looks like an unfilled template (do not reuse as final report)."""
    if not text or len(text) < 80:
        return True
    markers = (
        "[Number]",
        "[Percentage]",
        "[List",
        "[Description]",
        "[Value]",
        "[Target",
        "[Treatment",
        "[Imputation",
        "`[Number]`",
    )
    return any(m in text for m in markers)


REPORTER_GROUNDING_RULES = (
    "CRITICAL — report integrity:\n"
    "- Use ONLY facts from SESSION FACTS and COMPLETED STEPS below. Quote numbers and column names exactly as they appear there.\n"
    "- Forbidden: placeholder tokens like [Number], [Percentage], [List], [Description], [Value], or bracketed templates.\n"
    "- Forbidden: substituting a different dataset (e.g. Ames Housing / SalePrice, telco churn, customer_id) if this run is another dataset.\n"
    "- If the digest does not state a figure, write 'not detailed in prior steps' — do not invent statistics.\n"
    "- Start with '# Executive Summary' and keep markdown factual and specific to this run.\n"
)


# ============================================================================
# PART 5: WORKFLOW ORCHESTRATION WITH RETRIES
# ============================================================================

class DataAnalysisWorkflow:
    """Main workflow controller for sequential data analysis pipeline."""

    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "./analysis_results",
        resume_from: Optional[str] = None,
    ):
        self.dataset_path = str(Path(dataset_path).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resume_raw = (resume_from or os.getenv("RESUME_RUN_DIR", "") or "").strip()
        if not resume_raw:
            rid = (os.getenv("RESUME_RUN_ID") or os.getenv("RUN_ID") or "").strip()
            if rid:
                cand = self.output_dir / f"run_{rid}"
                if cand.is_dir():
                    resume_raw = str(cand)
        if resume_raw:
            self.run_output_dir = Path(resume_raw).resolve()
            self.run_output_dir.mkdir(parents=True, exist_ok=True)
            stem = self.run_output_dir.name
            self.run_id = stem[4:] if stem.startswith("run_") else datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_output_dir = self.output_dir / f"run_{self.run_id}"
            self.run_output_dir.mkdir(exist_ok=True)

        self.executor = PythonSessionTool(output_dir=str(self.run_output_dir / "charts"))
        self.executor.init_session(self.dataset_path)
        if resume_raw:
            if try_load_kernel_snapshot(self.executor, self.run_output_dir):
                print("[RESUME] Restored kernel state from kernel_snapshot/ (Parquet + meta.json).")
            else:
                print(
                    "[RESUME] No kernel_snapshot under this run folder — using CSV init only. "
                    "Save snapshots by running specialists (SESSION_SNAPSHOT=1, default)."
                )

        self.agents = create_agents(self.executor)
        self.tasks = create_tasks(self.agents)

        self.manager_chat_path = self.run_output_dir / "manager_chat.jsonl"
        self.session_meta_path = self.run_output_dir / "session_meta.json"
        self.run_history_path = self.run_output_dir / "run_history.json"
        self.manager_chat_records: List[Dict[str, Any]] = []
        self.run_history_dynamic: List[Dict[str, Any]] = []
        self.brief_text: str = ""
        self.brief_dict: Dict[str, Any] = {}

        if resume_raw and self.manager_chat_path.exists():
            self.manager_chat_records = load_jsonl_records(self.manager_chat_path)
        if resume_raw and self.run_history_path.exists():
            try:
                loaded = json.loads(self.run_history_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self.run_history_dynamic = loaded
            except Exception:
                self.run_history_dynamic = []

        self.results: Dict[str, Any] = {}
        self.charts: List[Path] = []
        self.report_path: Optional[Path] = None

    def _append_manager_chat_record(self, kind: str, role: str, content: str, **extra: Any) -> None:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "role": role,
            "content": content,
            **extra,
        }
        self.manager_chat_records.append(rec)
        persist_jsonl_record(self.manager_chat_path, rec)

    def _save_dynamic_run_history(self) -> None:
        self.run_history_path.write_text(
            json.dumps(self.run_history_dynamic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_kernel_snapshot_safe(self) -> None:
        try:
            save_kernel_snapshot(self.executor, self.run_output_dir)
        except Exception as e:
            print(f"[SESSION_SNAPSHOT] Save failed: {e}")

    def ensure_dynamic_brief(self) -> None:
        """Compute dataset brief for supervisor prompts if not already loaded."""
        if self.brief_text:
            return
        self.brief_text, self.brief_dict = compute_dataset_brief(self.executor, self.run_output_dir)

    def get_interactive_specialists(self) -> Dict[str, Agent]:
        if getattr(self, "_interactive_specialists_cache", None) is None:
            self._interactive_specialists_cache = create_dynamic_specialist_agents(self.executor)
        return self._interactive_specialists_cache

    def _next_specialist_step_index(self) -> int:
        if not self.run_history_dynamic:
            return 1
        return max(int(e.get("step", 0)) for e in self.run_history_dynamic) + 1

    def _session_facts_markdown(self) -> str:
        """Deterministic facts from the Python kernel for report grounding (not LLM-invented)."""
        g = self.executor.session_globals
        lines: List[str] = []
        dp = g.get("DATASET_PATH")
        if dp:
            lines.append(f"- Data file: {dp}")
        dr = g.get("df_raw")
        if isinstance(dr, pd.DataFrame):
            lines.append(f"- df_raw: {dr.shape[0]} rows × {dr.shape[1]} columns")
            lines.append(f"- df_raw columns: {', '.join(str(c) for c in dr.columns.tolist())}")
        dc = g.get("df_clean")
        if isinstance(dc, pd.DataFrame):
            lines.append(f"- df_clean: {dc.shape[0]} rows × {dc.shape[1]} columns")
        dfeat = g.get("df_features")
        if isinstance(dfeat, pd.DataFrame):
            lines.append(f"- df_features: {dfeat.shape[0]} rows × {dfeat.shape[1]} columns")
        nc = g.get("NUMERIC_COLUMNS")
        cc = g.get("CATEGORICAL_COLUMNS")
        if isinstance(nc, list):
            lines.append(f"- NUMERIC_COLUMNS ({len(nc)}): {', '.join(str(x) for x in nc[:40])}" + (" …" if len(nc) > 40 else ""))
        if isinstance(cc, list):
            lines.append(f"- CATEGORICAL_COLUMNS ({len(cc)}): {', '.join(str(x) for x in cc[:40])}" + (" …" if len(cc) > 40 else ""))
        vr = g.get("validation_report")
        if isinstance(vr, list) and vr:
            lines.append("- validation_report (sample entries):")
            for item in vr[:35]:
                lines.append(f"  - {str(item)[:480]}")
        elif isinstance(vr, dict) and vr:
            snippet = json.dumps(vr, ensure_ascii=False, default=str)[:3500]
            lines.append(f"- validation_report (JSON): {snippet}")
        lines.append(f"- TIME_INDEX_OK: {g.get('TIME_INDEX_OK')}")
        return "\n".join(lines) if lines else "(no session facts available)"

    def _run_dynamic_terminal_reporter(self, user_prompt: str, specialists: Dict[str, Agent]) -> None:
        """Synthesize markdown report from brief + run history (reuses last reporter specialist output when valid)."""
        hist_summary = format_run_history_digest(
            self.run_history_dynamic,
            last_n=80,
            max_instruction_chars=500,
            max_excerpt_chars=6000,
        )
        session_facts = self._session_facts_markdown()
        last_entry = self.run_history_dynamic[-1] if self.run_history_dynamic else None
        reused_reporter = False
        if last_entry and last_entry.get("agent") == "reporter":
            rstep = last_entry.get("step")
            tk_rep = f"dynamic_step_{rstep}_reporter"
            rep_text = str(self.results.get(tk_rep, "")).strip()
            if (rep_text.startswith("#") or "Executive Summary" in rep_text[:500]) and not _report_text_has_placeholders(
                rep_text
            ):
                self.results["report"] = rep_text
                reused_reporter = True
                print("[REPORT] Reusing output from last reporter specialist step.")
            elif rep_text and _report_text_has_placeholders(rep_text):
                print("[REPORT] Last reporter output had placeholders — regenerating with full context.")

        if not reused_reporter:
            report_desc = (
                f"USER GOAL:\n{user_prompt}\n\n"
                f"{REPORTER_GROUNDING_RULES}\n\n"
                f"DATASET BRIEF:\n{self.brief_text[:4000]}\n\n"
                f"SESSION FACTS:\n{session_facts}\n\n"
                f"RUN HISTORY (digest):\n{hist_summary}\n\n"
                "OUTPUT ONLY markdown. Start with '# Executive Summary'. Max ~800 words. No code blocks. "
                "Sections: Executive Summary, Data Overview, Quality & Cleaning, Key Findings, Statistical Highlights, "
                "Recommendations. Every statistic must match SESSION FACTS or RUN HISTORY excerpts above."
            )
            rep_task = Task(
                description=report_desc,
                expected_output="Markdown report.",
                agent=specialists["reporter"],
                async_execution=False,
            )
            try:
                rep_result = rep_task.execute_sync(agent=specialists["reporter"])
                self.results["report"] = str(rep_result)
            except Exception as e:
                print(f"[REPORT] Error: {e}")
                self.results["report"] = (
                    f"# Executive Summary\n\nReport generation failed: {e}\n\n## Run history\n{hist_summary[:6000]}"
                )

    def _dynamic_supervisor_single_turn(
        self,
        user_prompt: str,
        specialists: Dict[str, Agent],
        system_instruction: str,
        state: SupervisorLoopState,
        *,
        step_delay: float,
        max_specialist_steps: int,
        max_manager_turns: int,
        interactive_chat_breaks: bool,
        log: Optional[List[str]] = None,
    ) -> SupervisorSingleTurnOutcome:
        """One manager JSON decision: CHAT, DONE, guardrail stop, or one specialist Crew run."""

        def _out(s: str) -> None:
            if log is not None:
                log.append(s)
            else:
                print(s)

        state.manager_invocations += 1
        if max_manager_turns > 0 and state.manager_invocations > max_manager_turns:
            if interactive_chat_breaks:
                _out("[GUARDRAIL] INTERACTIVE_MAX_MANAGER_TURNS exceeded — stopping interactive loop.")
            else:
                _out("[GUARDRAIL] Stopping: INTERACTIVE_MAX_MANAGER_TURNS exceeded.")
            return "exit_manager_cap"

        flags = self.executor.validate_state()
        g = self.executor.session_globals
        df_clean_missing = g.get("df_clean") is None

        chat_block = build_manager_chat_block_for_llm(
            self.manager_chat_records,
            run_output_dir=self.run_output_dir,
        )
        digest = format_run_history_digest(self.run_history_dynamic)
        user_payload = (
            f"user_goal:\n{user_prompt}\n\n"
            f"validate_state:\n{json.dumps(flags)}\n\n"
            f"run_history_digest:\n{digest}\n"
        )
        if state.repeat_streak >= 2:
            user_payload += (
                "\nGUARDRAIL_HINT: The same specialist role was chosen repeatedly. "
                "Pick a different next_agent, narrow the instruction, use CHAT to explain, or DONE if satisfied.\n"
            )
        decision = invoke_manager_decision(
            chat_block=chat_block,
            user_payload=user_payload,
            system_instruction=system_instruction,
        )

        na = decision.next_agent
        if na not in ("DONE", "CHAT", "cleaning") and df_clean_missing:
            decision = ManagerDecision(
                next_agent="cleaning",
                instruction=(
                    "Run full preparation: env check, validation_report, and df_clean before other specialists."
                ),
                rationale="guardrail_df_clean_required",
            )
            na = decision.next_agent
        if (
            self.brief_dict.get("is_time_series")
            and decision.next_agent == "feature_engineering"
            and not g.get("TIME_INDEX_OK")
        ):
            decision = ManagerDecision(
                next_agent="cleaning",
                instruction=(
                    "Time-series: ensure df_clean is chronologically ordered by the time column; "
                    "set TIME_INDEX_OK = True in globals() when done."
                ),
                rationale="guardrail_time_index",
            )
            na = decision.next_agent

        self._append_manager_chat_record("message", "assistant", decision.model_dump_json())

        if decision.next_agent == "DONE":
            note = (decision.reply_to_user or "").strip()
            if note:
                _out(f"\n[MANAGER]\n{note}\n")
            return "break_interactive" if interactive_chat_breaks else "done"

        if decision.next_agent == "CHAT":
            reply = (decision.reply_to_user or "").strip() or (decision.rationale or "").strip()
            if reply:
                _out(f"\n[MANAGER]\n{reply}\n")
                self._append_manager_chat_record(
                    "chat_turn",
                    "assistant",
                    reply,
                    next_agent="CHAT",
                )
            state.repeat_streak = 0
            state.last_agent = None
            time.sleep(step_delay)
            return "break_interactive" if interactive_chat_breaks else "continue"

        if state.specialist_count >= max_specialist_steps:
            if interactive_chat_breaks:
                _out("[GUARDRAIL] DYNAMIC_MAX_STEPS (specialist runs) reached — type /report or exit.")
            else:
                _out("[GUARDRAIL] Stopping: DYNAMIC_MAX_STEPS (specialist runs) reached.")
            return "exit_specialist_cap"

        if decision.next_agent == state.last_agent:
            state.repeat_streak += 1
        else:
            state.repeat_streak = 0
        state.last_agent = decision.next_agent
        if state.repeat_streak >= 5:
            if interactive_chat_breaks:
                _out("[GUARDRAIL] Same agent repeated — stopping this turn.")
            else:
                _out("[GUARDRAIL] Stopping: same agent repeated without progress.")
            return "exit_repeat"

        state.specialist_count += 1
        step = self._next_specialist_step_index()
        use_ts = bool(self.brief_dict.get("is_time_series"))
        rep_facts: Optional[str] = None
        rep_hist: Optional[List[Dict[str, Any]]] = None
        if decision.next_agent == "reporter":
            rep_facts = self._session_facts_markdown()
            rep_hist = list(self.run_history_dynamic)
        task = build_specialist_task(
            agent_id=decision.next_agent,
            agents=specialists,
            user_prompt=user_prompt,
            manager_instruction=decision.instruction,
            use_ts_appendix=use_ts,
            reporter_session_facts=rep_facts,
            reporter_run_history=rep_hist,
        )
        task_key = f"dynamic_step_{step}_{decision.next_agent}"
        result = self._run_task_with_retry(
            agent=specialists[decision.next_agent],
            task=task,
            task_key=task_key,
            extra_inputs={},
        )
        excerpt = str(result)[:4500] if result is not None else ""
        post_flags = self.executor.validate_state()
        self.run_history_dynamic.append(
            {
                "step": step,
                "agent": decision.next_agent,
                "instruction": decision.instruction,
                "output_excerpt": excerpt,
                "state_flags": dict(post_flags),
            }
        )
        self._save_dynamic_run_history()
        self._append_manager_chat_record(
            "analysis_digest",
            "user",
            f"[Analysis step {step} | {decision.next_agent}]\n{excerpt}",
        )
        self._save_kernel_snapshot_safe()
        time.sleep(step_delay)
        return "continue"

    def run_dynamic_team_pipeline(
        self,
        user_prompt: str,
        max_steps: int = 18,
        followup_messages: Optional[List[str]] = None,
        *,
        skip_terminal_reporter: bool = False,
    ) -> Dict[str, Any]:
        """Supervisor loop: manager JSON routing + single-agent crews + chat/run JSON persistence.

        DYNAMIC_MAX_STEPS caps **specialist** Crew executions only; CHAT turns do not consume it.
        """
        max_specialist_steps = int(os.getenv("DYNAMIC_MAX_STEPS", str(max_steps)))
        step_delay = float(os.getenv("DYNAMIC_STEP_DELAY_SECONDS", "2"))
        mgr_cap_raw = os.getenv("INTERACTIVE_MAX_MANAGER_TURNS", "").strip()
        max_manager_turns = int(mgr_cap_raw) if mgr_cap_raw.isdigit() else 0

        print(f"\n{'='*70}")
        print("DYNAMIC SUPERVISOR PIPELINE (STATEFUL)")
        print(f"Dataset: {self.dataset_path}")
        print(f"Run ID: {self.run_id}")
        print(f"Output: {self.run_output_dir}")
        print(f"Max specialist steps: {max_specialist_steps}")
        if max_manager_turns:
            print(f"Max manager invocations (total): {max_manager_turns}")
        print(f"{'='*70}\n")

        self.brief_text, self.brief_dict = compute_dataset_brief(self.executor, self.run_output_dir)
        specialists = create_dynamic_specialist_agents(self.executor)

        meta = {
            "schema": 1,
            "mode": "dynamic_supervisor",
            "dataset_path": self.dataset_path,
            "run_id": self.run_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            self.session_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError:
            pass

        if not self.manager_chat_records:
            self._append_manager_chat_record(
                "message",
                "user",
                f"Initial user goal:\n{user_prompt}\n\nDataset brief:\n{self.brief_text[:8000]}",
            )

        _fus = list(followup_messages or [])
        _env_fu = os.getenv("USER_FOLLOWUP", "").strip()
        if _env_fu:
            _fus.append(_env_fu)
        for _txt in _fus:
            t = str(_txt).strip()
            if t:
                self._append_manager_chat_record("message", "user", f"User follow-up:\n{t}")

        system_instruction = build_manager_system_instruction(self.brief_dict)
        sup_state = SupervisorLoopState(specialist_count=len(self.run_history_dynamic))

        while True:
            turn = self._dynamic_supervisor_single_turn(
                user_prompt,
                specialists,
                system_instruction,
                sup_state,
                step_delay=step_delay,
                max_specialist_steps=max_specialist_steps,
                max_manager_turns=max_manager_turns,
                interactive_chat_breaks=False,
                log=None,
            )
            if turn == "continue":
                continue
            if turn == "done":
                break
            if turn in ("exit_manager_cap", "exit_specialist_cap", "exit_repeat"):
                break

        if not skip_terminal_reporter:
            self._run_dynamic_terminal_reporter(user_prompt, specialists)

        charts_dir = self.run_output_dir / "charts"
        if charts_dir.exists():
            self.charts = list(charts_dir.glob("*.png"))
        else:
            self.charts = []
        self.results["dynamic_run_history"] = list(self.run_history_dynamic)
        self._save_report_to_file()
        return self.results

    def _run_interactive_supervisor_segment(
        self,
        user_prompt: str,
        specialists: Dict[str, Agent],
        system_instruction: str,
        state: SupervisorLoopState,
        *,
        step_delay: float,
        max_specialist_steps: int,
        max_manager_turns: int,
        log: Optional[List[str]] = None,
    ) -> InteractiveSegmentResult:
        """Inner supervisor loop for one user message (until CHAT, DONE, or guardrail)."""
        while True:
            turn = self._dynamic_supervisor_single_turn(
                user_prompt,
                specialists,
                system_instruction,
                state,
                step_delay=step_delay,
                max_specialist_steps=max_specialist_steps,
                max_manager_turns=max_manager_turns,
                interactive_chat_breaks=True,
                log=log,
            )
            if turn == "exit_manager_cap":
                return InteractiveSegmentResult(outcome="session_exit_manager_cap")
            if turn == "continue":
                continue
            return InteractiveSegmentResult(outcome="await_user")

    def run_interactive_session(self, user_prompt: str) -> None:
        """Stdin loop: follow-up messages, CHAT/DONE/specialist routing. Terminal report: /report or on exit."""
        print(
            "\n[INTERACTIVE] Supervisor mode. Commands:  /report  = full markdown report;  exit | quit  = leave.\n"
        )
        specialists = create_dynamic_specialist_agents(self.executor)
        system_instruction = build_manager_system_instruction(self.brief_dict)
        step_delay = float(os.getenv("DYNAMIC_STEP_DELAY_SECONDS", "2"))
        max_specialist_steps = int(os.getenv("DYNAMIC_MAX_STEPS", "18"))
        mgr_cap_raw = os.getenv("INTERACTIVE_MAX_MANAGER_TURNS", "").strip()
        max_manager_turns = int(mgr_cap_raw) if mgr_cap_raw.isdigit() else 0

        sup_state = SupervisorLoopState(specialist_count=len(self.run_history_dynamic))

        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INTERACTIVE] End of input — exiting.")
                break

            if not line:
                continue
            low = line.lower()
            if low in ("exit", "quit", "q"):
                break
            if low == "/report":
                self._run_dynamic_terminal_reporter(user_prompt, specialists)
                self._save_report_to_file()
                print(f"[REPORT] Saved under {self.run_output_dir}")
                continue

            self._append_manager_chat_record("message", "user", line)

            seg = self._run_interactive_supervisor_segment(
                user_prompt,
                specialists,
                system_instruction,
                sup_state,
                step_delay=step_delay,
                max_specialist_steps=max_specialist_steps,
                max_manager_turns=max_manager_turns,
            )
            if seg.outcome == "session_exit_manager_cap":
                return

        rep_existing = str(self.results.get("report", "") or "").strip()
        if not rep_existing or len(rep_existing) < 80:
            self._run_dynamic_terminal_reporter(user_prompt, specialists)
        self._save_report_to_file()
        print(f"[INTERACTIVE] Session ended. Report: {self.run_output_dir}")


    def _run_task_with_retry(
        self,
        agent: Agent,
        task: Task,
        task_key: str,
        max_retries: int = 2,
        delay_seconds: int = 4,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run a Crew with retry on failure."""
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
                extra_inputs={},
            )
            print(f"[PHASE 2 - Task {i}/{len(analysis_order)}] [OK] Completed (or max retries reached)")

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
                "Generate a markdown report summarizing the data analysis results.\n\n"
                "=== ANALYSIS CONTEXT ===\n\n"
                "PREPARATION PHASE RESULTS:\n"
                f"{self.results.get('preparation', 'N/A')}\n\n"
                "ANALYSIS PHASE RESULTS:\n"
                f"{self.results.get('analysis', 'N/A')}\n\n"
                "=== OUTPUT INSTRUCTIONS ===\n"
                "OUTPUT ONLY the markdown report. Do NOT include any explanations, plans, or reasoning.\n"
                "Start directly with '# Executive Summary' - no preamble.\n\n"
                "=== REQUIRED SECTIONS ===\n"
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
            expected_output="Markdown report starting with '# Executive Summary' (<= 800 words). No preamble or explanations.",
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
            self.results["report"] = self._generate_fallback_report()

        charts_dir = self.run_output_dir / "charts"
        if charts_dir.exists():
            self.charts = list(charts_dir.glob("*.png"))
        else:
            self.charts = []
        
        self._save_report_to_file()
        
        return self.results
    
    def _generate_fallback_report(self) -> str:
        """Fallback report if LLM fails."""
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
        """Save report to run directory."""
        report_md = self.results.get("report", "")
        if not report_md or report_md.startswith("Error:"):
            report_md = self._generate_fallback_report()
        
        self.report_path = self.run_output_dir / f"analysis_report_{self.run_id}.md"
        try:
            self.report_path.write_text(report_md, encoding="utf-8")
            print(f"[REPORT] [OK] Report saved to: {self.report_path}")
        except Exception as e:
            print(f"[REPORT] [ERROR] Failed to save report: {e}")

    def generate_markdown_report(self) -> str:
        """Generate and save the markdown report."""
        report_path = self.report_path or (self.run_output_dir / f"analysis_report_{self.run_id}.md")
        
        if report_path.exists():
            print(f"\n{'='*70}")
            print("[OK] MARKDOWN REPORT ALREADY GENERATED")
            print(f"Location: {report_path}")
            print(f"Run directory: {self.run_output_dir}")
            print(f"{'='*70}\n")
            return str(report_path)
        
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
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    dataset_path = os.getenv("DATASET_PATH", "sample_data.csv")
    output_dir = os.getenv("OUTPUT_DIR", "./analysis_results")
    workflow_mode = os.getenv("WORKFLOW_MODE", "dynamic").strip().lower()
    user_prompt = os.getenv(
        "USER_ANALYSIS_PROMPT",
        "Perform exploratory analysis and preprocessing with clear recommendations.",
    ).strip()
    resume = os.getenv("RESUME_RUN_DIR", "").strip()

    if not Path(dataset_path).exists():
        print(f"Creating sample dataset: {dataset_path}")
        create_sample_dataset(dataset_path)

    workflow = DataAnalysisWorkflow(
        dataset_path=dataset_path,
        output_dir=output_dir,
        resume_from=resume or None,
    )

    if workflow_mode == "dynamic":
        workflow.run_dynamic_team_pipeline(user_prompt=user_prompt)
    else:
        workflow.run_sequential_pipeline()

    report_path = workflow.generate_markdown_report()

    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"Charts saved to: {workflow.run_output_dir / 'charts'}")

    return report_path


if __name__ == "__main__":
    main()
