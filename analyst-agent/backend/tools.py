import os
import uuid
import base64
import json
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from jupyter_client import KernelManager
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from nbclient import NotebookClient


class CodeInput(BaseModel):
    code: str = Field(description="Python code to execute")


class JupyterSessionTool(BaseTool):
    """
    Executes Python code in a persistent Jupyter IPython kernel backed by
    nbclient + nbformat.  Each call creates (or edits) a real notebook cell.
    All cells share the same kernel namespace — variables persist across calls.
    """

    name: str = "python_session"
    description: str = (
        "Execute Python code in a shared Jupyter kernel session. "
        "Variables persist between calls."
    )
    args_schema: type[BaseModel] = CodeInput

    _km: KernelManager | None = None
    _notebook: nbformat.NotebookNode | None = None
    _client: NotebookClient | None = None
    _execution_count: int = 0
    _output_dir: str = "./results/charts"
    _ws_callback: object | None = None
    _current_agent: str = "system"

    def __init__(self, output_dir: str = "./results/charts", ws_callback=None, **kwargs):
        super().__init__(**kwargs)
        self._km = None
        self._notebook = None
        self._client = None
        self._execution_count = 0
        self._output_dir = output_dir
        self._ws_callback = ws_callback
        self._current_agent = "system"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Kernel lifecycle
    # ------------------------------------------------------------------

    def start_kernel(self):
        """Start a new IPython kernel and nbclient session."""
        if self._km is not None:
            return
        self._km = KernelManager(kernel_name="python3")
        self._km.start_kernel()
        self._notebook = new_notebook()
        self._client = NotebookClient(
            nb=self._notebook,
            km=self._km,
            timeout=120,
            allow_errors=True,
        )
        self._client.kc = self._km.client()
        self._client.kc.start_channels()
        self._client.kc.wait_for_ready(timeout=30)

        self._execute_silent("%matplotlib inline")
        self._execute_silent(
            f"import os; os.makedirs(r'{self._output_dir}', exist_ok=True)"
        )

    def shutdown_kernel(self):
        """Shutdown the kernel. Safe to call multiple times."""
        try:
            if self._client and self._client.kc:
                self._client.kc.stop_channels()
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
        self._client = None
        self._notebook = None

    # ------------------------------------------------------------------
    # Silent execution (setup commands, not tracked as cells)
    # ------------------------------------------------------------------

    def _execute_silent(self, code: str):
        if not self._client or not self._client.kc:
            self.start_kernel()
        msg_id = self._client.kc.execute(code)
        while True:
            try:
                msg = self._client.kc.get_iopub_msg(timeout=30)
                if (
                    msg["parent_header"].get("msg_id") == msg_id
                    and msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except Exception:
                break

    # ------------------------------------------------------------------
    # Cell <-> record conversion helpers
    # ------------------------------------------------------------------

    def _ensure_cell_metadata(self, cell, agent_name: str) -> str:
        """Stamp cell.metadata with agent + cell_id if not already set."""
        if "cell_id" not in cell.metadata:
            self._execution_count += 1
            cell.metadata["cell_id"] = f"cell_{self._execution_count}_{uuid.uuid4().hex[:6]}"
            cell.metadata["execution_count"] = self._execution_count
        if "agent" not in cell.metadata:
            cell.metadata["agent"] = agent_name
        return cell.metadata["cell_id"]

    def _extract_outputs(self, cell) -> dict:
        """Convert nbformat cell outputs into our flat record format."""
        stdout_parts = []
        stderr_parts = []
        images = []
        agent_name = cell.metadata.get("agent", "system")
        exec_count = cell.metadata.get("execution_count", 0)

        for out in cell.outputs:
            otype = out.get("output_type", "")
            if otype == "stream":
                if out.get("name") == "stdout":
                    stdout_parts.append(out.get("text", ""))
                elif out.get("name") == "stderr":
                    stderr_parts.append(out.get("text", ""))
            elif otype in ("display_data", "execute_result"):
                data = out.get("data", {})
                if "image/png" in data:
                    img_name = f"{agent_name}_{exec_count}_{uuid.uuid4().hex[:6]}.png"
                    img_path = os.path.join(self._output_dir, img_name)
                    try:
                        img_bytes = base64.b64decode(data["image/png"])
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        images.append(img_path)
                    except Exception:
                        pass
                if "text/plain" in data:
                    stdout_parts.append(data["text/plain"])
            elif otype == "error":
                stderr_parts.append("\n".join(out.get("traceback", [])))

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        return {
            "cell_id": cell.metadata.get("cell_id", ""),
            "agent": agent_name,
            "code": cell.source,
            "stdout": stdout,
            "stderr": stderr,
            "images": images,
            "execution_count": exec_count,
            "success": not bool(stderr.strip()),
        }

    def _find_cell_index(self, cell_id: str) -> int | None:
        """Return the index of a cell by cell_id, or None."""
        if not self._notebook:
            return None
        for i, cell in enumerate(self._notebook.cells):
            if cell.metadata.get("cell_id") == cell_id:
                return i
        return None

    def _execute_cell_at(self, index: int) -> dict:
        """Execute the cell at the given index and return its record."""
        cell = self._notebook.cells[index]
        cell.outputs = []

        if not self._client or not self._client.kc:
            self.start_kernel()
        if self._km and not self._km.is_alive():
            self.shutdown_kernel()
            self.start_kernel()
            index = self._find_cell_index(cell.metadata.get("cell_id", ""))
            if index is None:
                return {"error": "Cell lost after kernel restart"}
            cell = self._notebook.cells[index]

        try:
            self._client.execute_cell(cell, index)
        except Exception as e:
            cell.outputs.append(nbformat.v4.new_output(
                output_type="stream", name="stderr",
                text=f"Execution error: {e}",
            ))

        return self._extract_outputs(cell)

    # ------------------------------------------------------------------
    # Agent-facing method (same interface as before)
    # ------------------------------------------------------------------

    def _run(self, code: str, agent_name: str = "system") -> str:
        """
        Execute code as a new cell in the Jupyter kernel.
        Returns JSON string with: stdout, stderr, images, success, cell_id
        """
        if not self._notebook or not self._client:
            self.start_kernel()

        if self._km and not self._km.is_alive():
            self.shutdown_kernel()
            self.start_kernel()

        if agent_name == "system" and self._current_agent != "system":
            agent_name = self._current_agent

        cell = new_code_cell(source=code)
        cell_id = self._ensure_cell_metadata(cell, agent_name)
        self._notebook.cells.append(cell)
        cell_index = len(self._notebook.cells) - 1

        record = self._execute_cell_at(cell_index)

        return json.dumps({
            "stdout": record["stdout"],
            "stderr": record["stderr"],
            "images": record["images"],
            "success": record["success"],
            "cell_id": cell_id,
            "execution_count": record["execution_count"],
        })

    # ------------------------------------------------------------------
    # Cell CRUD methods (shared by agents and users, full parity)
    # ------------------------------------------------------------------

    def create_cell(self, code: str, agent_name: str = "user", position: int | None = None) -> str:
        """Insert a new code cell without executing it. Returns cell_id."""
        if not self._notebook:
            self.start_kernel()

        cell = new_code_cell(source=code)
        cell_id = self._ensure_cell_metadata(cell, agent_name)

        if position is not None and 0 <= position <= len(self._notebook.cells):
            self._notebook.cells.insert(position, cell)
        else:
            self._notebook.cells.append(cell)

        return cell_id

    def edit_cell(self, cell_id: str, new_code: str) -> dict | None:
        """Edit a cell's source and clear its old outputs. Returns record or None."""
        idx = self._find_cell_index(cell_id)
        if idx is None:
            return None
        cell = self._notebook.cells[idx]
        cell.source = new_code
        cell.outputs = []
        return self._extract_outputs(cell)

    def delete_cell(self, cell_id: str) -> bool:
        """Remove a cell from the notebook. Returns True if found and removed."""
        idx = self._find_cell_index(cell_id)
        if idx is None:
            return False
        self._notebook.cells.pop(idx)
        return True

    def rerun_cell(self, cell_id: str) -> dict | None:
        """Re-execute an existing cell and return its updated record."""
        idx = self._find_cell_index(cell_id)
        if idx is None:
            return None
        return self._execute_cell_at(idx)

    def edit_and_rerun_cell(self, cell_id: str, new_code: str) -> dict | None:
        """Edit a cell's source, then re-execute it. Returns updated record."""
        idx = self._find_cell_index(cell_id)
        if idx is None:
            return None
        cell = self._notebook.cells[idx]
        cell.source = new_code
        cell.outputs = []
        return self._execute_cell_at(idx)

    def get_cells_by_agent(self, agent_name: str) -> list[str]:
        """Return all cell_ids belonging to the given agent."""
        if not self._notebook:
            return []
        return [
            cell.metadata["cell_id"]
            for cell in self._notebook.cells
            if cell.metadata.get("agent") == agent_name
        ]

    def export_notebook(self, path: str) -> str:
        """Write the in-memory notebook to disk as .ipynb. Returns the path."""
        if not self._notebook:
            raise ValueError("No notebook to export — kernel not started.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbformat.write(self._notebook, f)
        return str(p)

    # ------------------------------------------------------------------
    # Compatibility: get_cells returns flat records
    # ------------------------------------------------------------------

    def get_cells(self) -> list:
        """Return all cell records for this session (same format as before)."""
        if not self._notebook:
            return []
        return [self._extract_outputs(cell) for cell in self._notebook.cells]
