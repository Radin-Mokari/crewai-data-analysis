import os
import re
import uuid
import base64
import json
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from jupyter_client import KernelManager


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text (for clean traceback display)."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class CodeInput(BaseModel):
    code: str = Field(description="Python code to execute")


class JupyterSessionTool(BaseTool):
    """
    Executes Python code in a persistent Jupyter IPython kernel.
    Each call creates a new cell. All cells share the same kernel
    namespace — variables persist across calls.

    Pure jupyter_client implementation per CLAUDE.md spec:
    - KernelManager starts the kernel
    - KernelClient sends code via kc.execute()
    - Output collected by polling kc.get_iopub_msg()
    - Cells tracked as simple dicts in self._cells
    """

    name: str = "python_session"
    description: str = (
        "Execute Python code in a shared Jupyter kernel session. "
        "Variables persist between calls."
    )
    args_schema: type[BaseModel] = CodeInput

    # Instance state (not Pydantic fields)
    _km: KernelManager | None = None
    _kc: object | None = None  # KernelClient
    _cells: list = []
    _execution_count: int = 0
    _output_dir: str = "./results/charts"
    _ws_callback: object | None = None
    _current_agent: str = "system"

    def __init__(self, output_dir: str = "./results/charts", ws_callback=None, **kwargs):
        super().__init__(**kwargs)
        self._km = None
        self._kc = None
        self._cells = []
        self._execution_count = 0
        self._output_dir = output_dir
        self._ws_callback = ws_callback
        self._current_agent = "system"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Kernel lifecycle
    # ------------------------------------------------------------------

    def start_kernel(self):
        """Start a new IPython kernel with Google Colab-equivalent environment."""
        if self._km is not None:
            return  # Already running
        self._km = KernelManager(kernel_name="python3")
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)

        # --- Colab-equivalent setup ---
        # 1. Inline matplotlib rendering
        self._execute_silent("%matplotlib inline")
        # 2. Ensure output dir exists
        self._execute_silent(
            f"import os; os.makedirs(r'{self._output_dir}', exist_ok=True)"
        )
        # 3. Pre-load core data science libraries (like Colab)
        self._execute_silent("""import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
try:
    import seaborn as sns
    sns.set_theme(style='whitegrid')
except ImportError:
    pass
try:
    import scipy
    from scipy import stats
except ImportError:
    pass
try:
    import sklearn
except ImportError:
    pass
try:
    from IPython.display import display, HTML, Markdown, Image
except ImportError:
    pass
""")
        # 4. Set friendly pandas display options
        self._execute_silent("""
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
""")

    def shutdown_kernel(self):
        """Shutdown the kernel. Safe to call multiple times."""
        try:
            if self._kc:
                self._kc.stop_channels()
        except Exception:
            pass
        try:
            if self._km and self._km.is_alive():
                self._km.shutdown_kernel(now=True)
            elif self._km:
                self._km.cleanup_resources()
        except Exception:
            pass
        self._km = None
        self._kc = None

    # ------------------------------------------------------------------
    # Silent execution (setup commands, not tracked as cells)
    # ------------------------------------------------------------------

    def _execute_silent(self, code: str):
        """Execute code without tracking as a cell (for setup commands)."""
        if not self._kc:
            self.start_kernel()
        msg_id = self._kc.execute(code)
        # Wait for completion
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=30)
                if (
                    msg["parent_header"].get("msg_id") == msg_id
                    and msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except Exception:
                break

    # ------------------------------------------------------------------
    # Core IOPub message collector
    # ------------------------------------------------------------------

    def _collect_output(self, msg_id: str, agent_name: str) -> dict:
        """
        Collect all outputs from the IOPub channel for a given msg_id.
        Captures: stdout, stderr (clean), HTML tables, PNG/SVG images.
        Returns dict with stdout, stderr, html, images, svg, success.
        """
        stdout_parts = []
        stderr_parts = []
        html_parts = []
        svg_parts = []
        images = []

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=120)
            except Exception:
                stderr_parts.append("Timeout: kernel did not respond within 120s")
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                if content["name"] == "stdout":
                    stdout_parts.append(content["text"])
                elif content["name"] == "stderr":
                    # Strip ANSI escape codes for clean display
                    stderr_parts.append(_strip_ansi(content["text"]))

            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data", {})

                # PNG image (matplotlib, seaborn plots)
                if "image/png" in data:
                    img_name = f"{agent_name}_{self._execution_count}_{uuid.uuid4().hex[:6]}.png"
                    img_path = os.path.join(self._output_dir, img_name)
                    try:
                        img_bytes = base64.b64decode(data["image/png"])
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        images.append(img_path)
                    except Exception:
                        pass

                # SVG image
                elif "image/svg+xml" in data:
                    svg_parts.append(data["image/svg+xml"])

                # HTML output (pandas DataFrames, display(HTML(...)), etc.)
                if "text/html" in data:
                    html_parts.append(data["text/html"])

                # Plain text fallback (only if no richer representation)
                if "text/plain" in data and "text/html" not in data and "image/png" not in data:
                    stdout_parts.append(data["text/plain"])

            elif msg_type == "error":
                # Clean ANSI codes from tracebacks
                tb = "\n".join(
                    _strip_ansi(line)
                    for line in content.get("traceback", [])
                )
                stderr_parts.append(tb)

            elif msg_type == "status" and content["execution_state"] == "idle":
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        html = "".join(html_parts)
        svg = "".join(svg_parts)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "html": html,
            "svg": svg,
            "images": images,
            "success": not bool(stderr.strip()),
        }

    # ------------------------------------------------------------------
    # Agent-facing method (CrewAI tool interface)
    # ------------------------------------------------------------------

    def _run(self, code: str, agent_name: str = "system") -> str:
        """
        Execute code as a new cell in the Jupyter kernel.
        Returns JSON string with: stdout, stderr, images, success, cell_id
        """
        if not self._kc:
            self.start_kernel()

        # Restart kernel if it died
        if self._km and not self._km.is_alive():
            self.shutdown_kernel()
            self.start_kernel()

        # Inherit current agent name if caller didn't specify
        if agent_name == "system" and self._current_agent != "system":
            agent_name = self._current_agent

        self._execution_count += 1
        cell_id = f"cell_{self._execution_count}_{uuid.uuid4().hex[:6]}"

        msg_id = self._kc.execute(code)
        output = self._collect_output(msg_id, agent_name)

        # Build cell record
        cell = {
            "cell_id": cell_id,
            "agent": agent_name,
            "code": code,
            "stdout": output["stdout"],
            "stderr": output["stderr"],
            "html": output["html"],
            "svg": output["svg"],
            "images": output["images"],
            "execution_count": self._execution_count,
            "success": output["success"],
        }
        self._cells.append(cell)

        return json.dumps({
            "stdout": output["stdout"],
            "stderr": output["stderr"],
            "html": output["html"],
            "svg": output["svg"],
            "images": output["images"],
            "success": output["success"],
            "cell_id": cell_id,
            "execution_count": self._execution_count,
        })

    # ------------------------------------------------------------------
    # Cell query methods
    # ------------------------------------------------------------------

    def get_cells(self) -> list:
        """Return all cell records for this session."""
        return self._cells.copy()

    def get_cells_by_agent(self, agent_name: str) -> list[str]:
        """Return all cell_ids belonging to the given agent."""
        return [
            cell["cell_id"]
            for cell in self._cells
            if cell.get("agent") == agent_name
        ]

    # ------------------------------------------------------------------
    # Cell CRUD methods (for REST API / user interaction)
    # ------------------------------------------------------------------

    def _find_cell(self, cell_id: str) -> tuple[int, dict] | tuple[None, None]:
        """Return (index, cell_dict) for a cell_id, or (None, None)."""
        for i, cell in enumerate(self._cells):
            if cell["cell_id"] == cell_id:
                return i, cell
        return None, None

    def create_cell(self, code: str, agent_name: str = "user", position: int | None = None) -> str:
        """Insert a new code cell without executing it. Returns cell_id."""
        self._execution_count += 1
        cell_id = f"cell_{self._execution_count}_{uuid.uuid4().hex[:6]}"
        cell = {
            "cell_id": cell_id,
            "agent": agent_name,
            "code": code,
            "stdout": "",
            "stderr": "",
            "html": "",
            "svg": "",
            "images": [],
            "execution_count": self._execution_count,
            "success": True,
        }
        if position is not None and 0 <= position <= len(self._cells):
            self._cells.insert(position, cell)
        else:
            self._cells.append(cell)
        return cell_id

    def edit_cell(self, cell_id: str, new_code: str) -> dict | None:
        """Edit a cell's source and clear its old outputs. Returns record or None."""
        idx, cell = self._find_cell(cell_id)
        if cell is None:
            return None
        cell["code"] = new_code
        cell["stdout"] = ""
        cell["stderr"] = ""
        cell["html"] = ""
        cell["svg"] = ""
        cell["images"] = []
        cell["success"] = True
        return cell

    def delete_cell(self, cell_id: str) -> bool:
        """Remove a cell from the list. Returns True if found and removed."""
        idx, cell = self._find_cell(cell_id)
        if idx is None:
            return False
        self._cells.pop(idx)
        return True

    def rerun_cell(self, cell_id: str) -> dict | None:
        """Re-execute an existing cell and return its updated record."""
        idx, cell = self._find_cell(cell_id)
        if cell is None:
            return None

        if not self._kc:
            self.start_kernel()
        if self._km and not self._km.is_alive():
            self.shutdown_kernel()
            self.start_kernel()

        msg_id = self._kc.execute(cell["code"])
        output = self._collect_output(msg_id, cell["agent"])

        cell["stdout"] = output["stdout"]
        cell["stderr"] = output["stderr"]
        cell["html"] = output["html"]
        cell["svg"] = output["svg"]
        cell["images"] = output["images"]
        cell["success"] = output["success"]
        return cell

    def edit_and_rerun_cell(self, cell_id: str, new_code: str) -> dict | None:
        """Edit a cell's source, then re-execute it. Returns updated record."""
        idx, cell = self._find_cell(cell_id)
        if cell is None:
            return None

        cell["code"] = new_code

        if not self._kc:
            self.start_kernel()
        if self._km and not self._km.is_alive():
            self.shutdown_kernel()
            self.start_kernel()

        msg_id = self._kc.execute(new_code)
        output = self._collect_output(msg_id, cell["agent"])

        cell["stdout"] = output["stdout"]
        cell["stderr"] = output["stderr"]
        cell["html"] = output["html"]
        cell["svg"] = output["svg"]
        cell["images"] = output["images"]
        cell["success"] = output["success"]
        return cell

    # ------------------------------------------------------------------
    # Export — rebuild .ipynb from cell records
    # ------------------------------------------------------------------

    def export_notebook(self, path: str) -> str:
        """Write the cell history to disk as a .ipynb notebook."""
        import nbformat
        from nbformat.v4 import new_notebook, new_code_cell

        nb = new_notebook()
        for cell in self._cells:
            nb_cell = new_code_cell(source=cell["code"])
            nb_cell.metadata["agent"] = cell.get("agent", "system")
            nb_cell.metadata["cell_id"] = cell["cell_id"]
            # Add outputs
            outputs = []
            if cell.get("stdout"):
                outputs.append(nbformat.v4.new_output(
                    output_type="stream", name="stdout", text=cell["stdout"]
                ))
            if cell.get("stderr"):
                outputs.append(nbformat.v4.new_output(
                    output_type="stream", name="stderr", text=cell["stderr"]
                ))
            # Add HTML output (DataFrames, display(HTML(...)), etc.)
            if cell.get("html"):
                outputs.append(nbformat.v4.new_output(
                    output_type="execute_result",
                    data={"text/html": cell["html"], "text/plain": ""},
                    execution_count=cell.get("execution_count", 1),
                ))
            # Add SVG output
            if cell.get("svg"):
                outputs.append(nbformat.v4.new_output(
                    output_type="display_data",
                    data={"image/svg+xml": cell["svg"], "text/plain": "<SVG>"},
                ))
            # Add PNG images
            for img_path in cell.get("images", []):
                try:
                    with open(img_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    outputs.append(nbformat.v4.new_output(
                        output_type="display_data",
                        data={"image/png": img_b64, "text/plain": "<Figure>"},
                    ))
                except Exception:
                    pass
            nb_cell.outputs = outputs
            nb.cells.append(nb_cell)

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return str(p)
