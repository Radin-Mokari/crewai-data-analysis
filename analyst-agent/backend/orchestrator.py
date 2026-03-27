"""
AnalysisOrchestrator — core orchestration with Manager ReAct loop.
Supports Path A (full analysis) and Path B (single specialist).
"""

import asyncio
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from crewai import Agent, Task, Crew, Process

from backend.tools import JupyterSessionTool


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
            "errors": [],
            "redo_count": 0,
            "agent_redo_count": {},
        }
        self._known_cell_count = 0
        self._specialist_cells: dict[str, list[str]] = {}

    # -----------------------------------------------------------------------
    # Path A — Full Analysis (Manager-driven)
    # -----------------------------------------------------------------------

    async def run(self, dataset_path: str, user_prompt: str) -> dict:
        try:
            self.tool.start_kernel()
            await self._emit("progress", "Starting Jupyter kernel...")

            await self._emit("progress", "Inspecting dataset...")
            self.state["profile"] = await asyncio.to_thread(inspect_dataset, self.tool, dataset_path)
            await self._emit("progress", f"Dataset loaded: {self.state['profile'].get('shape', '?')}")
            await self._emit_new_cells()

            max_iterations = 10
            for iteration in range(max_iterations):
                snapshot = self._build_snapshot(user_prompt)
                await self._emit("agent_thought", f"[Manager] OBSERVE (iteration {iteration + 1}):\n{snapshot[:500]}")

                decision_text = await asyncio.to_thread(self._ask_manager, snapshot)
                await self._emit("agent_thought", f"[Manager] DECIDE: {decision_text}")

                decision = self._parse_decision(decision_text)

                if decision["action"] == "COMPLETE":
                    await self._emit("progress", "Analysis complete!")
                    break

                if decision["action"] == "DELEGATE":
                    specialist_name = decision["specialist"]
                    instructions = decision["instructions"]

                    # Guard: don't re-run a specialist that already passed quality
                    if specialist_name in self.state["completed"]:
                        await self._emit("agent_thought", f"[Manager] Skipping {specialist_name} — already completed.")
                        await self._emit("progress", f"Skipping {specialist_name} (already done)")
                        continue

                    await self._emit("progress", f"Delegating to {specialist_name}...")

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
        finally:
            self.tool.shutdown_kernel()

        self._save_report()

        return self.state

    # -----------------------------------------------------------------------
    # Path B — Direct Single Agent
    # -----------------------------------------------------------------------

    async def run_single_specialist(self, dataset_path: str, agent_name: str, user_prompt: str) -> dict:
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

    def _build_snapshot(self, user_prompt: str) -> str:
        profile = self.state.get("profile") or {}
        lines = [
            f"USER REQUEST: {user_prompt}",
            f"DATASET: {profile.get('shape', '?')}",
            f"COLUMNS: {profile.get('columns', [])}",
            f"NUMERIC: {profile.get('numeric_columns', [])}",
            f"CATEGORICAL: {profile.get('categorical_columns', [])}",
            f"MISSING VALUES: {profile.get('missing_values', {})}",
            f"DUPLICATES: {profile.get('duplicates', 0)}",
            f"COMPLETED TASKS: {self.state['completed']}",
        ]
        if self.state["task_summaries"]:
            lines.append("TASK SUMMARIES:")
            for name, summary in self.state["task_summaries"].items():
                lines.append(f"  {name}: {summary[:200]}")
        if self.state["errors"]:
            lines.append(f"RECENT ERRORS: {self.state['errors'][-3:]}")
        return "\n".join(lines)

    def _ask_manager(self, snapshot: str) -> str:
        task_description = (
            f"Based on the current analysis state below, decide the next action.\n\n"
            f"{snapshot}\n\n"
            f"Respond with exactly one line: DELEGATE:specialist_name | instructions\n"
            f"or: COMPLETE"
        )
        task = Task(
            description=task_description,
            expected_output="One line: DELEGATE:name | instructions OR COMPLETE",
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
        # Extract just the last DELEGATE or COMPLETE line from potentially verbose output
        for line in reversed(decision.strip().splitlines()):
            line = line.strip()
            if line.upper() == "COMPLETE":
                return {"action": "COMPLETE"}
            match = re.search(r"DELEGATE:\s*(\w+)\s*\|\s*(.+)", line, re.IGNORECASE)
            if match:
                specialist = match.group(1).strip().lower()
                instructions = match.group(2).strip()
                if specialist in self.VALID_SPECIALISTS:
                    return {"action": "DELEGATE", "specialist": specialist, "instructions": instructions}

        if "COMPLETE" in decision.upper():
            return {"action": "COMPLETE"}

        for name in self.VALID_SPECIALISTS:
            if name in decision.lower():
                return {"action": "DELEGATE", "specialist": name, "instructions": decision}

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
            report_context_parts = []
            for sname, summary in self.state["task_summaries"].items():
                report_context_parts.append(f"=== {sname.upper()} ===\n{summary[:3000]}")
            report_context = "\n\n".join(report_context_parts) if report_context_parts else "No prior analysis."

            task_desc = (
                f"PRIOR ANALYSIS FINDINGS:\n{report_context}\n\n"
                f"DATASET PROFILE: {json.dumps(self.state.get('profile', {}), default=str)[:1500]}\n\n"
                f"CHART FILES: {self.state.get('charts', [])}\n\n"
                f"INSTRUCTIONS: {instructions}\n\n"
                f"{redo_context}"
                f"Write a comprehensive markdown report synthesizing ALL the findings above.\n"
                f"Every section above contains real data — use specific numbers, column names, and findings.\n"
                f"The report MUST include all sections: Dataset Overview, Data Quality, EDA, Visualizations, Statistical Analysis, Key Findings."
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
            return str(result).strip()
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
        """Extract clean report markdown from task_summaries."""
        report_text = self.state["task_summaries"].get("report", "")
        if not report_text:
            return ""
        if "AGENT SUMMARY:" in report_text:
            report_text = report_text.split("AGENT SUMMARY:", 1)[1].strip()
        return report_text

    def _save_report(self):
        report_text = self._get_report_text()
        if not report_text:
            return

        output_dir = Path(self.tool._output_dir).parent
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"report_{timestamp}.md"
        report_path.write_text(report_text, encoding="utf-8")
        self.state["report_path"] = str(report_path)

    def save_results_bundle(self, target_base: str) -> dict:
        """
        Save report + all chart images into a timestamped subdirectory
        under target_base (e.g. analysis_results/).
        Returns {"run_dir": str, "report_path": str, "charts_copied": int}.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(target_base) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        report_text = self._get_report_text()
        report_path = None
        if report_text:
            report_path = run_dir / f"analysis_report_{timestamp}.md"
            report_path.write_text(report_text, encoding="utf-8")

        charts_copied = 0
        for chart_path_str in self.state.get("charts", []):
            src = Path(chart_path_str)
            if src.exists():
                dst = run_dir / src.name
                shutil.copy2(src, dst)
                charts_copied += 1

        return {
            "run_dir": str(run_dir),
            "report_path": str(report_path) if report_path else None,
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

    def _evaluate_quality(self, specialist_name: str, result: str, cells_before: int = 0) -> bool:
        if not result:
            return False

        all_cells = self.tool.get_cells()
        specialist_cells = [
            c for c in all_cells[cells_before:]
            if c.get("agent") == specialist_name
        ]

        for cell in specialist_cells:
            stderr = cell.get("stderr", "")
            if self._is_real_error(stderr):
                return False

        if len(result.strip()) <= 20:
            return False

        if specialist_name != "report":
            try:
                check_result = self.tool._run(KERNEL_CHECK_CODE, agent_name="validation")
                check_data = json.loads(check_result)
                if "DataFrame" not in check_data.get("stdout", ""):
                    return False
            except Exception:
                return False

        return True

    # -----------------------------------------------------------------------
    # WebSocket Emission
    # -----------------------------------------------------------------------

    async def _emit(self, event_type: str, content):
        if self.ws:
            await self.ws.broadcast({
                "type": event_type,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            })

    async def _emit_new_cells(self):
        """Emit cell_update events for any new or changed cells since last check."""
        all_cells = self.tool.get_cells()
        if len(all_cells) > self._known_cell_count:
            for cell in all_cells[self._known_cell_count:]:
                await self._emit("cell_update", cell)
        else:
            for cell in all_cells:
                await self._emit("cell_update", cell)
        self._known_cell_count = len(all_cells)
