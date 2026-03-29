"""
AnalysisOrchestrator — core orchestration with Manager ReAct loop.
Supports Path A (full analysis) and Path B (single specialist).
"""

import asyncio
import json
import logging
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from crewai import Agent, Task, Crew, Process

from backend.tools import JupyterSessionTool

# Configure logging
logger = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Inspection Phase (deterministic, no LLM)
# ---------------------------------------------------------------------------

def inspect_dataset(tool: JupyterSessionTool, dataset_path: str) -> dict:
    """
    Run two cells in the Jupyter kernel to load and profile the dataset.
    Returns the profile dict.
    """
    load_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df_raw = pd.read_csv(r'{dataset_path}')
df_clean = None
df_features = None
DATASET_COLUMNS = list(df_raw.columns)
NUMERIC_COLUMNS = df_raw.select_dtypes(include=[np.number]).columns.tolist()
CATEGORICAL_COLUMNS = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
DATASET_SHAPE = df_raw.shape
ORIGINAL_NUMERIC_COLUMNS = NUMERIC_COLUMNS.copy()
ORIGINAL_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS.copy()
print(f"Loaded {{DATASET_SHAPE[0]}} rows x {{DATASET_SHAPE[1]}} columns")
"""
    tool._run(load_code, agent_name="inspection")

    profile_code = """
import json as _json

# Compute correlations for numeric columns
_corr_matrix = df_raw[NUMERIC_COLUMNS].corr() if len(NUMERIC_COLUMNS) > 1 else None
_top_correlations = []
if _corr_matrix is not None:
    for i, col1 in enumerate(NUMERIC_COLUMNS):
        for col2 in NUMERIC_COLUMNS[i+1:]:
            corr_val = abs(_corr_matrix.loc[col1, col2])
            if corr_val > 0.5:
                _top_correlations.append((col1, col2, round(corr_val, 3)))
    _top_correlations.sort(key=lambda x: x[2], reverse=True)
    _top_correlations = _top_correlations[:10]  # Top 10

# Compute skewness for numeric columns
_skewness = {}
for col in NUMERIC_COLUMNS:
    try:
        _skewness[col] = round(float(df_raw[col].skew()), 3)
    except:
        pass
_highly_skewed = [col for col, skew in _skewness.items() if abs(skew) > 1]

# Identify binary columns
_binary_columns = [col for col in df_raw.columns if df_raw[col].nunique() == 2]

# Identify potential target columns (binary or low cardinality categorical)
_potential_targets = [col for col in df_raw.columns if 2 <= df_raw[col].nunique() <= 10]

# Check class balance for potential targets (for class_imbalance skip decision)
_class_balance = {}
for col in _binary_columns + [c for c in _potential_targets if c not in _binary_columns]:
    try:
        counts = df_raw[col].value_counts(normalize=True)
        if len(counts) >= 2:
            majority_pct = float(counts.iloc[0] * 100)
            minority_pct = float(counts.iloc[1] * 100)
            _class_balance[col] = {
                'majority_pct': round(majority_pct, 1),
                'minority_pct': round(minority_pct, 1),
                'is_imbalanced': bool(majority_pct > 70)
            }
    except:
        pass

_profile = {
    'shape': list(df_raw.shape),
    'columns': list(df_raw.columns),
    'dtypes': df_raw.dtypes.astype(str).to_dict(),
    'numeric_columns': NUMERIC_COLUMNS,
    'categorical_columns': CATEGORICAL_COLUMNS,
    'missing_values': df_raw.isnull().sum().to_dict(),
    'missing_pct': (df_raw.isnull().sum() / len(df_raw) * 100).round(1).to_dict(),
    'duplicates': int(df_raw.duplicated().sum()),
    'describe': df_raw.describe().round(2).to_dict(),
    'top_correlations': _top_correlations,
    'highly_skewed_columns': _highly_skewed,
    'skewness': _skewness,
    'binary_columns': _binary_columns,
    'potential_targets': _potential_targets,
    'class_balance': _class_balance,
}
print(_json.dumps(_profile))
"""
    result_json = tool._run(profile_code, agent_name="inspection")
    result = json.loads(result_json)
    stdout = result.get("stdout", "")

    try:
        profile = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    profile = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            profile = {"shape": [], "columns": [], "error": "Could not parse profile"}

    return profile


# ---------------------------------------------------------------------------
# State Validation Code (run as a cell before each specialist)
# ---------------------------------------------------------------------------

VALIDATION_CODE = """
_available = {}
for _name in ['df_raw', 'df_clean', 'df_features']:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, 'shape'):
        _available[_name] = list(_obj.shape)
print(f"Available DataFrames: {_available}")
print(f"NUMERIC_COLUMNS: {globals().get('NUMERIC_COLUMNS', 'NOT SET')}")
print(f"CATEGORICAL_COLUMNS: {globals().get('CATEGORICAL_COLUMNS', 'NOT SET')}")
"""

KERNEL_CHECK_CODE = "print(type(df_raw).__name__)"

ERROR_SIGNALS = [
    "Traceback", "Exception", "KeyError", "TypeError",
    "ValueError", "NameError", "AttributeError", "IndexError",
]

HARMLESS_STDERR_PATTERNS = [
    "UserWarning",
    "FigureCanvasAgg is non-interactive",
    "FixedFormatter",
    "FixedLocator",
    "More than 20 figures have been opened",
    "Glyph",
    "FutureWarning",
    "DeprecationWarning",
]


# ---------------------------------------------------------------------------
# AnalysisOrchestrator
# ---------------------------------------------------------------------------

class AnalysisOrchestrator:
    VALID_SPECIALISTS = [
        "cleaning", "eda", "visualization", "statistics",
        "feature_engineering", "class_imbalance", "report",
    ]

    def __init__(self, tool, specialists, manager, llm, ws_manager=None):
        self.tool: JupyterSessionTool = tool
        self.specialists: dict[str, Agent] = specialists
        self.manager: Agent = manager
        self.llm = llm
        self.ws = ws_manager
        self.state = {
            "profile": None,
            "completed": [],
            "task_summaries": {},
            "charts": [],
            "chart_descriptions": [],  # Track what each chart shows
            "errors": [],
            "redo_count": 0,
            "agent_redo_count": {},
            "report_text": "",
            "quality_issues": {},  # Track quality issues per specialist
            "improvement_attempts": {},  # Track improvement attempts
        }
        self._known_cell_count = 0
        self._cell_lock = threading.Lock()  # Protect cell tracking from race conditions
        self._specialist_cells: dict[str, list[str]] = {}
        self._skipped_this_run: list[str] = []  # Track agents skipped due to already-completed
        self._event_count = 0  # Track number of events emitted for debugging

    # -----------------------------------------------------------------------
    # Path A — Full Analysis (Manager-driven)
    # -----------------------------------------------------------------------

    async def run(self, dataset_path: str, user_prompt: str, messages: list = None) -> dict:
        try:
            self.tool.start_kernel()
            await self._emit("progress", "Starting Jupyter kernel...")

            await self._emit("progress", "Inspecting dataset...")
            self.state["profile"] = await asyncio.to_thread(inspect_dataset, self.tool, dataset_path)
            await self._emit("progress", f"Dataset loaded: {self.state['profile'].get('shape', '?')}")
            await self._emit_new_cells()

            max_iterations = 10
            self._skipped_this_run = []  # Reset at start of each analysis
            for iteration in range(max_iterations):
                snapshot = self._build_snapshot(user_prompt, self._skipped_this_run, messages)
                await self._emit("agent_thought", f"[Manager] OBSERVE (iteration {iteration + 1}):\n{snapshot[:500]}")

                decision_text = await asyncio.to_thread(self._ask_manager, snapshot)
                await self._emit("agent_thought", f"[Manager] DECIDE: {decision_text}")

                decision = self._parse_decision(decision_text)

                if decision["action"] == "COMPLETE":
                    # Enforce that report must be completed before we can truly complete
                    if "report" not in self.state["completed"]:
                        await self._emit("agent_thought", "[Manager] Cannot COMPLETE without running report. Auto-delegating to report...")
                        # Force delegation to report
                        decision = {
                            "action": "DELEGATE",
                            "specialist": "report",
                            "instructions": "Synthesize all findings into a comprehensive markdown report.",
                            "is_improvement": False,
                        }
                    else:
                        await self._emit("progress", "Analysis complete!")
                        break

                if decision["action"] == "DELEGATE":
                    specialist_name = decision["specialist"]
                    instructions = decision["instructions"]
                    is_improvement = decision.get("is_improvement", False)

                    # Guard: limit improvement attempts to 2 per specialist
                    if specialist_name in self.state["completed"] and not is_improvement:
                        await self._emit("agent_thought", f"[Manager] Skipping {specialist_name} — already completed. Must pick a different agent.")
                        await self._emit("progress", f"Skipping {specialist_name} (already done)")
                        # Track this so the next snapshot tells Manager to pick something else
                        if specialist_name not in self._skipped_this_run:
                            self._skipped_this_run.append(specialist_name)
                        continue

                    if is_improvement:
                        attempts = self.state["improvement_attempts"].get(specialist_name, 0)
                        if attempts >= 2:
                            await self._emit("agent_thought", f"[Manager] Max improvement attempts (2) reached for {specialist_name}. Moving on.")
                            # Mark this specialist as improvement-exhausted so Manager knows to proceed
                            if "improvement_exhausted" not in self.state:
                                self.state["improvement_exhausted"] = []
                            if specialist_name not in self.state["improvement_exhausted"]:
                                self.state["improvement_exhausted"].append(specialist_name)
                            # Clear visualization gaps since we can't improve further
                            if specialist_name == "visualization":
                                self.state["visualization_coverage"] = {}
                                if "visualization" in self.state.get("quality_issues", {}):
                                    del self.state["quality_issues"]["visualization"]
                            continue
                        self.state["improvement_attempts"][specialist_name] = attempts + 1
                        await self._emit("progress", f"Improving {specialist_name} (attempt {attempts + 1}/2)...")

                    await self._emit("progress", f"Delegating to {specialist_name}...")
                    # Clear skipped list since we're actually running an agent
                    self._skipped_this_run = []

                    prior_cell_ids = self._specialist_cells.get(specialist_name)

                    cells_before = len(self.tool.get_cells())
                    result = await asyncio.to_thread(
                        self._run_specialist, specialist_name, instructions,
                        prior_cell_ids=prior_cell_ids,
                    )

                    self._collect_charts_from(cells_before)
                    await self._emit_new_cells()
                    self._track_specialist_cells(specialist_name)

                    summary = self._build_summary(specialist_name, result, cells_before)
                    self.state["task_summaries"][specialist_name] = summary

                    passed = self._evaluate_quality(specialist_name, result, cells_before)
                    if passed:
                        if specialist_name not in self.state["completed"]:
                            self.state["completed"].append(specialist_name)
                        await self._emit("quality_event", f"[PASS] {specialist_name}")
                    else:
                        self.state["redo_count"] += 1
                        agent_redos = self.state["agent_redo_count"].get(specialist_name, 0) + 1
                        self.state["agent_redo_count"][specialist_name] = agent_redos
                        if agent_redos >= 2:
                            # Per-agent cap reached — force-pass and move on
                            if specialist_name not in self.state["completed"]:
                                self.state["completed"].append(specialist_name)
                            await self._emit("quality_event", f"[FORCE-PASS] {specialist_name} (max 2 redos/agent)")
                        else:
                            if specialist_name in self.state["task_summaries"]:
                                del self.state["task_summaries"][specialist_name]
                            await self._emit("quality_event", f"[FAIL] {specialist_name} (redo {agent_redos}/2)")
                        if self.state["redo_count"] >= 5:
                            await self._emit("progress", "Max redos reached, completing...")
                            break
            else:
                await self._emit("progress", "Max iterations reached, completing...")

        except Exception as e:
            self.state["errors"].append(str(e))
            await self._emit("progress", f"Error: {str(e)}")
            raise

        self._save_report()

        return self.state

    # -----------------------------------------------------------------------
    # Path B — Direct Single Agent
    # -----------------------------------------------------------------------

    async def run_single_specialist(self, dataset_path: str, agent_name: str, user_prompt: str, messages: list = None) -> dict:
        if agent_name not in self.VALID_SPECIALISTS:
            raise ValueError(f"Unknown specialist: {agent_name}")

        try:
            if self.tool._km is None:
                self.tool.start_kernel()
                await self._emit("progress", "Starting Jupyter kernel...")

            if self.state["profile"] is None:
                await self._emit("progress", "Inspecting dataset...")
                self.state["profile"] = await asyncio.to_thread(inspect_dataset, self.tool, dataset_path)
                await self._emit_new_cells()

            await self._emit("progress", f"Running {agent_name}...")
            cells_before = len(self.tool.get_cells())
            result = await asyncio.to_thread(self._run_specialist, agent_name, user_prompt)

            self._collect_charts_from(cells_before)
            await self._emit_new_cells()
            self._track_specialist_cells(agent_name)

            summary = self._build_summary(agent_name, result, cells_before)
            self.state["task_summaries"][agent_name] = summary

            passed = self._evaluate_quality(agent_name, result, cells_before)
            if not passed:
                await self._emit("quality_event", f"[FAIL] {agent_name} — retrying with edit-in-place...")
                prior_cell_ids = self._specialist_cells.get(agent_name)
                cells_before_retry = len(self.tool.get_cells())
                result = await asyncio.to_thread(
                    self._run_specialist, agent_name, user_prompt,
                    prior_cell_ids=prior_cell_ids,
                )
                self._collect_charts_from(cells_before_retry)
                await self._emit_new_cells()
                self._track_specialist_cells(agent_name)
                summary = self._build_summary(agent_name, result, cells_before_retry)
                self.state["task_summaries"][agent_name] = summary
                passed = self._evaluate_quality(agent_name, result, cells_before_retry)

            if passed:
                if agent_name not in self.state["completed"]:
                    self.state["completed"].append(agent_name)
                await self._emit("quality_event", f"[PASS] {agent_name}")
            else:
                if agent_name not in self.state["completed"]:
                    self.state["completed"].append(agent_name)
                await self._emit("quality_event", f"[FORCE-PASS] {agent_name}")

            await self._emit("progress", f"{agent_name} complete.")

            if agent_name == "report":
                self._save_report()

        except Exception as e:
            self.state["errors"].append(str(e))
            await self._emit("progress", f"Error in {agent_name}: {str(e)}")
            raise

        return self.state

    # -----------------------------------------------------------------------
    # Manager Helpers
    # -----------------------------------------------------------------------

    def _build_snapshot(self, user_prompt: str, skipped_agents: list[str] | None = None, messages: list = None) -> str:
        profile = self.state.get("profile") or {}
        shape = profile.get('shape', [0, 0])
        n_rows = shape[0] if len(shape) > 0 else 0

        # Calculate missing and duplicate percentages for skip decisions
        missing_values = profile.get('missing_values', {})
        total_missing = sum(missing_values.values())
        duplicates = profile.get('duplicates', 0)
        dup_pct = (duplicates / n_rows * 100) if n_rows > 0 else 0

        lines = []

        if messages:
            history = "\n".join([f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')}" for m in messages[-5:]])
            lines.append(f"--- RECENT CONVERSATION HISTORY ---\n{history}\n----------------------------------\n")

        # CRITICAL: If agents were just skipped, tell the Manager FIRST
        if skipped_agents:
            remaining = [a for a in self.VALID_SPECIALISTS if a not in self.state["completed"]]
            lines.append("=" * 50)
            lines.append(f"⚠️  YOUR LAST DELEGATION TO {skipped_agents} WAS BLOCKED!")
            lines.append(f"    Those agents are ALREADY COMPLETED.")
            lines.append(f"    You MUST pick from: {remaining}")
            lines.append(f"    Or say COMPLETE if the analysis is done.")
            lines.append("=" * 50)
            lines.append("")

        lines.extend([
            f"CURRENT USER REQUEST: {user_prompt}",
            f"DATASET: {shape} ({n_rows} rows)",
            f"COLUMNS: {profile.get('columns', [])}",
            f"NUMERIC: {profile.get('numeric_columns', [])}",
            f"CATEGORICAL: {profile.get('categorical_columns', [])}",
        ])

        # Cleaning skip criteria — make it obvious
        lines.append("\n--- DATA QUALITY (for cleaning skip decision) ---")
        lines.append(f"TOTAL MISSING VALUES: {total_missing}")
        lines.append(f"DUPLICATES: {duplicates} ({dup_pct:.1f}% of rows)")
        if total_missing == 0 and dup_pct < 1:
            lines.append("→ DATA IS CLEAN: No missing values, duplicates < 1%")

        # Add data characteristics (for manager to use when delegating)
        lines.append("\n--- DATA CHARACTERISTICS ---")
        top_corr = profile.get('top_correlations', [])
        if top_corr:
            lines.append(f"TOP CORRELATIONS (|r|>0.5): {top_corr[:5]}")
        skewed = profile.get('highly_skewed_columns', [])
        if skewed:
            lines.append(f"HIGHLY SKEWED COLUMNS: {skewed}")
        binary = profile.get('binary_columns', [])
        if binary:
            lines.append(f"BINARY COLUMNS: {binary}")
        targets = profile.get('potential_targets', [])
        if targets:
            lines.append(f"POTENTIAL TARGETS: {targets}")

        # Class balance info for class_imbalance skip decision
        class_balance = profile.get('class_balance', {})
        if class_balance:
            imbalanced = [col for col, info in class_balance.items() if info.get('is_imbalanced')]
            balanced = [col for col, info in class_balance.items() if not info.get('is_imbalanced')]
            if imbalanced:
                lines.append(f"IMBALANCED COLUMNS (>70/30): {imbalanced}")
            if balanced:
                lines.append(f"BALANCED COLUMNS (<70/30): {balanced}")
            if not imbalanced:
                lines.append("→ NO CLASS IMBALANCE: All potential targets are balanced")

        lines.append(f"\n--- ANALYSIS STATUS ---")
        lines.append(f"COMPLETED TASKS: {self.state['completed']}")

        # Guide the Manager on what to do next
        completed = self.state['completed']
        viz_coverage = self.state.get("visualization_coverage", {})
        viz_gaps = viz_coverage.get("gaps", [])
        improvement_exhausted = self.state.get("improvement_exhausted", [])

        # Show exhausted improvements prominently
        if improvement_exhausted:
            lines.append(f"\n⚠️ IMPROVEMENT EXHAUSTED FOR: {improvement_exhausted}")
            lines.append("   Cannot improve further. Proceed to next step.")

        # Case 1: Both EDA and visualization done → ready for report
        if "visualization" in completed and "report" not in completed:
            # If visualization improvements exhausted, don't suggest more improvements
            if "visualization" in improvement_exhausted:
                lines.append("→ READY FOR REPORT: Visualization complete (max improvements reached). Delegate to 'report' next.")
            elif viz_gaps:
                lines.append("→ VISUALIZATION GAPS DETECTED: Use IMPROVE:visualization to add missing charts before report.")
            else:
                lines.append("→ READY FOR REPORT: Visualizations created. Delegate to 'report' next.")
        # Case 2: EDA done but not visualization → do visualization
        elif "eda" in completed and "visualization" not in completed:
            lines.append("→ NEXT STEP: EDA done. Delegate to 'visualization' with specific chart instructions.")
        # Case 3: Nothing done yet → start with eda
        elif "eda" not in completed and "visualization" not in completed:
            lines.append("→ NEXT STEP: Start with 'eda' to explore the data.")

        # Include chart descriptions if visualization was completed
        if self.state.get("chart_descriptions"):
            lines.append("\nCHARTS CREATED:")
            for desc in self.state["chart_descriptions"]:
                lines.append(f"  - {desc}")

        # Include visualization coverage analysis for Manager to evaluate
        viz_coverage = self.state.get("visualization_coverage")
        if viz_coverage:
            lines.append("\nVISUALIZATION COVERAGE ANALYSIS:")
            lines.append(f"  Correlation charts: {'YES' if viz_coverage.get('has_correlation_viz') else 'NO'}")
            lines.append(f"  Distribution charts: {'YES' if viz_coverage.get('has_distribution_viz') else 'NO'}")
            lines.append(f"  Categorical charts: {'YES' if viz_coverage.get('has_categorical_viz') else 'NO'}")
            lines.append(f"  Columns visualized: {viz_coverage.get('visualized_columns', [])}")
            if viz_coverage.get("gaps"):
                lines.append("  ⚠️ COVERAGE GAPS (consider using IMPROVE:visualization):")
                for gap in viz_coverage["gaps"]:
                    lines.append(f"    - {gap}")

        # Include quality issues found
        if self.state.get("quality_issues"):
            lines.append("\nQUALITY ISSUES FOUND:")
            for agent, issues in self.state["quality_issues"].items():
                lines.append(f"  {agent}: {issues}")

        # Include fuller task summaries (500 chars instead of 200)
        if self.state["task_summaries"]:
            lines.append("\nTASK SUMMARIES:")
            for name, summary in self.state["task_summaries"].items():
                lines.append(f"  {name}: {summary[:500]}")

        if self.state["errors"]:
            lines.append(f"\nRECENT ERRORS: {self.state['errors'][-3:]}")

        return "\n".join(lines)

    def _ask_manager(self, snapshot: str) -> str:
        task_description = (
            f"CURRENT STATE:\n{snapshot}\n\n"
            f"RULES:\n"
            f"- NEVER delegate to an agent already in COMPLETED TASKS\n"
            f"- Progress forward: eda → visualization → statistics → report → COMPLETE\n"
            f"- Output ONLY one line, no explanation\n\n"
            f"YOUR RESPONSE (one line only):"
        )
        task = Task(
            description=task_description,
            expected_output="DELEGATE:agent_name | instructions  OR  COMPLETE",
            agent=self.manager,
        )
        crew = Crew(
            agents=[self.manager],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        return str(result).strip()

    def _parse_decision(self, decision: str) -> dict:
        # Extract just the last DELEGATE, IMPROVE, or COMPLETE line from potentially verbose output
        for line in reversed(decision.strip().splitlines()):
            line = line.strip()
            if line.upper() == "COMPLETE":
                return {"action": "COMPLETE"}

            # Check for IMPROVE: action (re-delegate with improvements)
            improve_match = re.search(r"IMPROVE:\s*(\w+)\s*\|\s*(.+)", line, re.IGNORECASE)
            if improve_match:
                specialist = improve_match.group(1).strip().lower()
                instructions = improve_match.group(2).strip()
                if specialist in self.VALID_SPECIALISTS:
                    return {"action": "DELEGATE", "specialist": specialist, "instructions": instructions, "is_improvement": True}

            # Check for regular DELEGATE: action
            match = re.search(r"DELEGATE:\s*(\w+)\s*\|\s*(.+)", line, re.IGNORECASE)
            if match:
                specialist = match.group(1).strip().lower()
                instructions = match.group(2).strip()
                if specialist in self.VALID_SPECIALISTS:
                    return {"action": "DELEGATE", "specialist": specialist, "instructions": instructions, "is_improvement": False}

        if "COMPLETE" in decision.upper():
            return {"action": "COMPLETE"}

        for name in self.VALID_SPECIALISTS:
            if name in decision.lower():
                return {"action": "DELEGATE", "specialist": name, "instructions": decision, "is_improvement": False}

        return {"action": "COMPLETE"}

    # -----------------------------------------------------------------------
    # Specialist Cell Tracking (for edit-in-place on redo)
    # -----------------------------------------------------------------------

    def _track_specialist_cells(self, name: str):
        """Record which cell_ids belong to a specialist (for redo edits)."""
        self._specialist_cells[name] = self.tool.get_cells_by_agent(name)

    # -----------------------------------------------------------------------
    # Specialist Execution (Context Injection)
    # -----------------------------------------------------------------------

    def _build_report_context(self) -> str:
        """
        Build comprehensive context for the report agent using FULL cell outputs.
        This captures actual code executed and outputs produced by each specialist.
        """
        all_cells = self.tool.get_cells()
        specialist_outputs = {}

        # Group cells by agent (excluding system/validation/inspection cells)
        for cell in all_cells:
            agent = cell.get("agent", "system")
            if agent in ["system", "validation", "inspection"]:
                continue
            if agent not in specialist_outputs:
                specialist_outputs[agent] = []
            specialist_outputs[agent].append(cell)

        context_parts = []
        for specialist_name in self.state.get("completed", []):
            cells = specialist_outputs.get(specialist_name, [])
            if not cells:
                continue

            section = [f"=== {specialist_name.upper()} ANALYSIS ==="]

            for i, cell in enumerate(cells, 1):
                code = cell.get("code", "").strip()
                stdout = cell.get("stdout", "").strip()
                images = cell.get("images", [])

                section.append(f"\n--- Cell {i} ---")
                if code:
                    # Include full code (up to 2000 chars per cell)
                    section.append(f"CODE:\n```python\n{code[:2000]}\n```")
                if stdout:
                    # Include full output (up to 3000 chars per cell)
                    section.append(f"OUTPUT:\n{stdout[:3000]}")
                if images:
                    section.append(f"CHARTS GENERATED: {images}")

            context_parts.append("\n".join(section))

        return "\n\n".join(context_parts) if context_parts else "No prior analysis completed."

    def _run_specialist(self, name: str, instructions: str, prior_cell_ids: list[str] | None = None) -> str:
        agent = self.specialists[name]

        self.tool._current_agent = name

        prior_parts = []
        for sname, summary in self.state["task_summaries"].items():
            prior_parts.append(f"[{sname}]: {summary[:600]}")
        prior_context = "\n".join(prior_parts) if prior_parts else "No prior analysis completed."

        validation_output = ""
        if name != "report":
            val_result = self.tool._run(VALIDATION_CODE, agent_name="validation")
            val_data = json.loads(val_result)
            validation_output = val_data.get("stdout", "")

        redo_context = ""
        if prior_cell_ids:
            redo_context = (
                f"\nREDO MODE: Your previous attempt failed quality checks.\n"
                f"Your previous cell IDs: {prior_cell_ids}\n"
                f"Fix your code and produce correct results.\n"
            )

        if name == "report":
            # Use comprehensive context with full cell outputs instead of truncated summaries
            report_context = self._build_report_context()

            task_desc = (
                f"PRIOR ANALYSIS FINDINGS (with actual code and outputs):\n{report_context}\n\n"
                f"DATASET PROFILE: {json.dumps(self.state.get('profile', {}), default=str)}\n\n"
                f"ALL CHART FILES GENERATED: {self.state.get('charts', [])}\n\n"
                f"INSTRUCTIONS: {instructions}\n\n"
                f"{redo_context}"
                f"Write a comprehensive markdown report synthesizing ALL the findings above.\n"
                f"USE THE ACTUAL NUMBERS, COLUMN NAMES, AND STATISTICS from the cell outputs.\n"
                f"The report MUST include these sections:\n"
                f"1. Executive Summary\n"
                f"2. Dataset Overview (shape, columns, types)\n"
                f"3. Data Quality & Cleaning (what was fixed)\n"
                f"4. Exploratory Data Analysis (correlations, distributions, patterns)\n"
                f"5. Visualizations (reference the chart files by name)\n"
                f"6. Statistical Analysis (test results, significance)\n"
                f"7. Key Findings & Recommendations\n"
                f"Be DETAILED and SPECIFIC - include actual values from the analysis."
            )
        else:
            task_desc = (
                f"CORE MODE: Use existing variables in the shared Jupyter kernel.\n"
                f"Do NOT call pd.read_csv() — data is already loaded.\n"
                f"Use the best available DataFrame: df_features > df_clean > df_raw.\n\n"
                f"AVAILABLE KERNEL STATE:\n{validation_output}\n\n"
                f"PRIOR ANALYSIS FINDINGS:\n{prior_context}\n\n"
                f"{redo_context}"
                f"INSTRUCTIONS: {instructions}"
            )

        task = Task(
            description=task_desc,
            expected_output="Summary of what was done (2-3 sentences).",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            raw_result = str(result).strip()

            # Store full report text separately — not subject to summary truncation
            if name == "report":
                self.state["report_text"] = raw_result

            return raw_result
        except Exception as e:
            error_msg = f"Specialist {name} failed: {str(e)}"
            self.state["errors"].append(error_msg)
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(5)
                try:
                    result = crew.kickoff()
                    return str(result).strip()
                except Exception as e2:
                    self.state["errors"].append(f"Retry failed: {str(e2)}")
                    return error_msg
            return error_msg

    # -----------------------------------------------------------------------
    # Summary Builder — captures real cell outputs, not just agent's answer
    # -----------------------------------------------------------------------

    def _build_summary(self, specialist_name: str, agent_answer: str, cells_before: int) -> str:
        all_cells = self.tool.get_cells()
        new_cells = [
            c for c in all_cells[cells_before:]
            if c.get("agent") == specialist_name
        ]

        cell_outputs = []
        for cell in new_cells:
            stdout = (cell.get("stdout") or "").strip()
            if stdout and len(stdout) > 10:
                cell_outputs.append(stdout)

        parts = []
        if cell_outputs:
            combined_output = "\n".join(cell_outputs)
            parts.append(f"CELL OUTPUTS:\n{combined_output[:800]}")
        if agent_answer and len(agent_answer.strip()) > 10:
            parts.append(f"AGENT SUMMARY: {agent_answer[:300]}")

        summary = "\n".join(parts) if parts else (agent_answer or "No output captured.")
        return summary[:500]

    # -----------------------------------------------------------------------
    # Chart Collection — runs BEFORE _emit_new_cells updates the counter
    # -----------------------------------------------------------------------

    def _collect_charts_from(self, cells_before: int):
        """Collect chart image paths from cells created since cells_before."""
        for cell in self.tool.get_cells()[cells_before:]:
            for img in cell.get("images", []):
                if img not in self.state["charts"]:
                    self.state["charts"].append(img)

    # -----------------------------------------------------------------------
    # Report Saving
    # -----------------------------------------------------------------------

    def _get_report_text(self) -> str:
        """Extract clean report markdown — uses full text, not truncated summary."""
        # Prefer the full stored report text (set in _run_specialist when name == "report")
        report_text = self.state.get("report_text", "")

        if report_text:
            # We have the full report - return it as-is
            return report_text.strip()

        # Fallback to task_summaries if report_text wasn't stored
        fallback = self.state["task_summaries"].get("report", "")
        if not fallback:
            return ""

        # The fallback from task_summaries has "AGENT SUMMARY:" prefix from _build_summary()
        if "AGENT SUMMARY:" in fallback:
            fallback = fallback.split("AGENT SUMMARY:", 1)[1].strip()

        return fallback

    def _save_report(self):
        report_text = self._get_report_text()

        output_dir = Path(self.tool._output_dir).parent
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save report markdown (if available)
        if report_text:
            report_path = report_dir / f"report_{timestamp}.md"
            report_path.write_text(report_text, encoding="utf-8")
            self.state["report_path"] = str(report_path)

        # Always save the notebook (captures all cell execution history)
        try:
            notebook_path = report_dir / f"notebook_{timestamp}.ipynb"
            self.tool.export_notebook(str(notebook_path))
            self.state["notebook_path"] = str(notebook_path)
        except Exception as e:
            self.state["errors"].append(f"Notebook export failed: {str(e)}")

    def save_results_bundle(self, target_base: str) -> dict:
        """
        Save report + notebook + all chart images into a timestamped subdirectory
        under target_base (e.g. analysis_results/).
        Structure:
          analysis_results/run_<timestamp>/
            charts/
              chart_xxx.png
            analysis_report_<timestamp>.md
            analysis_notebook_<timestamp>.ipynb
        Returns {"run_dir": str, "report_path": str, "notebook_path": str, "charts_copied": int}.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(target_base) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create charts subdirectory
        charts_dir = run_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        report_text = self._get_report_text()
        report_path = None
        if report_text:
            report_path = run_dir / f"analysis_report_{timestamp}.md"
            report_path.write_text(report_text, encoding="utf-8")

        # Save notebook (export all cells with outputs)
        notebook_path = None
        try:
            notebook_path = run_dir / f"analysis_notebook_{timestamp}.ipynb"
            self.tool.export_notebook(str(notebook_path))
        except Exception as e:
            # Don't fail the whole bundle if notebook export fails
            self.state["errors"].append(f"Notebook export failed: {str(e)}")
            notebook_path = None

        # Copy charts to charts subdirectory
        charts_copied = 0
        for chart_path_str in self.state.get("charts", []):
            src = Path(chart_path_str)
            if src.exists():
                dst = charts_dir / src.name
                shutil.copy2(src, dst)
                charts_copied += 1

        return {
            "run_dir": str(run_dir),
            "report_path": str(report_path) if report_path else None,
            "notebook_path": str(notebook_path) if notebook_path else None,
            "charts_copied": charts_copied,
        }

    # -----------------------------------------------------------------------
    # Quality Gate
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_real_error(stderr: str) -> bool:
        """Check if stderr contains a real error vs. just harmless warnings."""
        if not stderr.strip():
            return False
        for line in stderr.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            is_harmless = any(pat in line_stripped for pat in HARMLESS_STDERR_PATTERNS)
            if is_harmless:
                continue
            for signal in ERROR_SIGNALS:
                if signal in line_stripped:
                    return True
        return False

    def _extract_chart_descriptions(self, cells: list) -> list[str]:
        """Extract descriptions of what charts were created from cell code."""
        descriptions = []
        chart_keywords = {
            "scatter": "scatter plot",
            "hist": "histogram",
            "bar": "bar chart",
            "box": "box plot",
            "heatmap": "heatmap",
            "corr": "correlation",
            "pie": "pie chart",
            "line": "line plot",
            "kde": "density plot",
            "violin": "violin plot",
            "pairplot": "pair plot",
            "countplot": "count plot",
            "regplot": "regression plot",
        }

        for cell in cells:
            code = cell.get("code", "").lower()
            images = cell.get("images", [])

            if images:
                # Try to extract what was plotted
                desc_parts = []

                # Check chart type
                chart_type = "chart"
                for keyword, name in chart_keywords.items():
                    if keyword in code:
                        chart_type = name
                        break

                # Try to extract column names from code
                # Look for patterns like df['column'], df.column, x='column', y='column'
                col_pattern = r"(?:df(?:_\w+)?)\[[\'\"](\w+)[\'\"]\]|(?:x|y|hue|data)=[\'\"](\w+)[\'\"]"
                matches = re.findall(col_pattern, code)
                cols = [m[0] or m[1] for m in matches if m[0] or m[1]]

                if cols:
                    desc_parts.append(f"{chart_type} of {', '.join(cols[:3])}")
                else:
                    desc_parts.append(chart_type)

                for img in images:
                    img_name = Path(img).name
                    descriptions.append(f"{desc_parts[0]} ({img_name})")

        return descriptions

    def _check_visualization_coverage(self, cells: list) -> dict:
        """
        Check if visualizations cover key data characteristics.
        Returns dict with coverage status and gaps.
        """
        profile = self.state.get("profile", {})

        # What the data has
        top_correlations = profile.get("top_correlations", [])
        highly_skewed = profile.get("highly_skewed_columns", [])
        binary_cols = profile.get("binary_columns", [])
        class_balance = profile.get("class_balance", {})
        imbalanced_cols = [col for col, info in class_balance.items() if info.get("is_imbalanced")]

        # Track what was visualized
        visualized_cols = set()
        has_correlation_viz = False
        has_distribution_viz = False
        has_categorical_viz = False

        chart_keywords_correlation = ["heatmap", "corr", "scatter", "regplot", "pairplot"]
        chart_keywords_distribution = ["hist", "box", "violin", "kde", "distplot"]
        chart_keywords_categorical = ["countplot", "bar", "pie"]

        for cell in cells:
            code = cell.get("code", "").lower()
            images = cell.get("images", [])

            if not images:
                continue

            # Check chart types
            for kw in chart_keywords_correlation:
                if kw in code:
                    has_correlation_viz = True
                    break
            for kw in chart_keywords_distribution:
                if kw in code:
                    has_distribution_viz = True
                    break
            for kw in chart_keywords_categorical:
                if kw in code:
                    has_categorical_viz = True
                    break

            # Extract visualized columns
            col_pattern = r"(?:df(?:_\w+)?)\[[\'\"](\w+)[\'\"]\]|(?:x|y|hue|data)=[\'\"](\w+)[\'\"]"
            matches = re.findall(col_pattern, code)
            for m in matches:
                col = m[0] or m[1]
                if col:
                    visualized_cols.add(col)

        # Determine gaps
        gaps = []

        # Check correlation coverage
        if top_correlations and not has_correlation_viz:
            corr_cols = set()
            for c1, c2, _ in top_correlations[:3]:
                corr_cols.add(c1)
                corr_cols.add(c2)
            gaps.append(f"Missing correlation visualization for highly correlated columns: {list(corr_cols)[:4]}")

        # Check skewness coverage
        if highly_skewed:
            skewed_not_viz = [c for c in highly_skewed[:5] if c not in visualized_cols]
            if skewed_not_viz and not has_distribution_viz:
                gaps.append(f"Missing distribution plots for skewed columns: {skewed_not_viz}")

        # Check binary/imbalanced coverage
        if imbalanced_cols:
            imbalanced_not_viz = [c for c in imbalanced_cols if c not in visualized_cols]
            if imbalanced_not_viz and not has_categorical_viz:
                gaps.append(f"Missing count/bar plots for imbalanced columns: {imbalanced_not_viz}")

        coverage = {
            "has_correlation_viz": has_correlation_viz,
            "has_distribution_viz": has_distribution_viz,
            "has_categorical_viz": has_categorical_viz,
            "visualized_columns": list(visualized_cols),
            "gaps": gaps,
            "is_sufficient": len(gaps) == 0,
        }

        return coverage

    def _evaluate_output_quality(self, specialist_name: str, cells: list) -> tuple[bool, list[str]]:
        """
        Basic quality check - did the agent produce meaningful output?
        The MANAGER will evaluate if it matches the requested analysis.
        Returns (passed, issues_list).
        """
        issues = []

        # Check: At least some output was produced
        all_output = " ".join(cell.get("stdout", "") for cell in cells)
        chart_count = sum(len(cell.get("images", [])) for cell in cells)

        if specialist_name == "visualization":
            if chart_count == 0:
                issues.append("No charts were generated")
                return False, issues
            # Record chart count for manager to see
            self.state["quality_issues"][specialist_name] = [f"Created {chart_count} chart(s)"]

        elif specialist_name == "eda":
            if len(all_output) < 100:
                issues.append("EDA output too brief")
                return False, issues

        # Manager will evaluate if the actual content matches expectations
        return True, issues

    def _evaluate_quality(self, specialist_name: str, result: str, cells_before: int = 0) -> bool:
        """
        Evaluate quality of specialist output.
        Checks both technical correctness (no errors) and semantic meaningfulness.
        """
        if not result:
            self.state["quality_issues"][specialist_name] = ["No output produced"]
            return False

        all_cells = self.tool.get_cells()
        specialist_cells = [
            c for c in all_cells[cells_before:]
            if c.get("agent") == specialist_name
        ]

        # Check 1: No runtime errors
        for cell in specialist_cells:
            stderr = cell.get("stderr", "")
            if self._is_real_error(stderr):
                self.state["quality_issues"][specialist_name] = [f"Runtime error: {stderr[:200]}"]
                return False

        # Check 2: Output is not trivially short
        if len(result.strip()) <= 20:
            self.state["quality_issues"][specialist_name] = ["Output too short/trivial"]
            return False

        # Check 3: Kernel state is valid (except for report agent)
        if specialist_name != "report":
            try:
                check_result = self.tool._run(KERNEL_CHECK_CODE, agent_name="validation")
                check_data = json.loads(check_result)
                if "DataFrame" not in check_data.get("stdout", ""):
                    self.state["quality_issues"][specialist_name] = ["DataFrame not found in kernel"]
                    return False
            except Exception:
                self.state["quality_issues"][specialist_name] = ["Kernel state validation failed"]
                return False

        # Check 4: Basic output quality (Manager will do semantic evaluation)
        if specialist_name in ["visualization", "eda"]:
            # Extract chart descriptions for manager context
            if specialist_name == "visualization":
                chart_descs = self._extract_chart_descriptions(specialist_cells)
                self.state["chart_descriptions"] = chart_descs

                # Check visualization coverage
                coverage = self._check_visualization_coverage(specialist_cells)
                self.state["visualization_coverage"] = coverage

                # Store gaps for Manager to see
                if coverage["gaps"]:
                    self.state["quality_issues"][specialist_name] = coverage["gaps"]

            # Basic quality check - did they produce output?
            passed, issues = self._evaluate_output_quality(specialist_name, specialist_cells)
            if not passed:
                self.state["quality_issues"][specialist_name] = issues
                return False  # Hard fail if no output at all

        # Clear any previous quality issues if we passed (unless there are coverage gaps)
        if specialist_name in self.state["quality_issues"]:
            # Don't clear visualization coverage gaps - Manager needs to see them
            if specialist_name != "visualization" or not self.state.get("visualization_coverage", {}).get("gaps"):
                del self.state["quality_issues"][specialist_name]

        return True

    # -----------------------------------------------------------------------
    # WebSocket Emission
    # -----------------------------------------------------------------------

    async def _emit(self, event_type: str, content):
        self._event_count += 1
        event = {
            "type": event_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        if self.ws:
            logger.info(f"[Emit #{self._event_count}] {event_type}: {str(content)[:100]}...")
            await self.ws.broadcast(event)
        else:
            logger.warning(f"[Emit #{self._event_count}] No WebSocket manager - {event_type} event lost")

    async def _emit_new_cells(self):
        """Emit cell_update events for any new or changed cells since last check."""
        with self._cell_lock:
            all_cells = self.tool.get_cells()
            current_count = len(all_cells)

            if current_count > self._known_cell_count:
                new_cells = all_cells[self._known_cell_count:]
                logger.info(f"[Cells] Emitting {len(new_cells)} new cells (total: {current_count})")
                for cell in new_cells:
                    agent = cell.get("agent", "unknown")
                    cell_id = cell.get("cell_id", "unknown")
                    logger.debug(f"[Cells] Emitting cell {cell_id} from {agent}")
                    await self._emit("cell_update", cell)
            else:
                logger.debug(f"[Cells] No new cells to emit (known: {self._known_cell_count}, current: {current_count})")

            self._known_cell_count = current_count
