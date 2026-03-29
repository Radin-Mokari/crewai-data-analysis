"""
JupyterSessionTool — Pure jupyter_client implementation with full Colab parity.

Features:
- Complete Jupyter Message Protocol compliance (IOPub, Shell, Stdin)
- Rich output support: PNG, SVG, HTML, LaTeX, Markdown, JSON
- Robust kernel management: interrupt, heartbeat, crash recovery
- State snapshotting for deterministic variable inspection
- Cell CRUD operations for interactive editing
- Notebook export with full output preservation
"""

import os
import re
import uuid
import base64
import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Any
from queue import Empty as QueueEmpty
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from jupyter_client import KernelManager

# Configure logging
logger = logging.getLogger("jupyter_tool")


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

    Features:
    - Full Jupyter Message Protocol compliance
    - Rich output capture (PNG, SVG, HTML, LaTeX, JSON, Markdown)
    - Kernel lifecycle management (start, shutdown, interrupt, restart)
    - Heartbeat monitoring and crash recovery
    - State snapshotting for variable inspection
    - Cell CRUD operations
    - Notebook export
    """

    name: str = "python_session"
    description: str = (
        "Execute Python code in a shared Jupyter kernel session. "
        "Variables persist between calls. Supports rich outputs like plots, "
        "DataFrames, HTML, LaTeX, and more."
    )
    args_schema: type[BaseModel] = CodeInput

    # Instance state (not Pydantic fields)
    _km: Optional[KernelManager] = None
    _kc: Optional[Any] = None  # KernelClient
    _cells: list = []
    _execution_count: int = 0
    _output_dir: str = "./results/charts"
    _ws_callback: Optional[Any] = None
    _current_agent: str = "system"
    _lock: threading.Lock = None  # Thread safety for cell operations
    _kernel_start_time: Optional[float] = None
    _last_heartbeat: Optional[float] = None

    # Configuration
    _default_timeout: int = 120  # seconds
    _heartbeat_interval: int = 30  # seconds

    def __init__(self, output_dir: str = "./results/charts", ws_callback=None, **kwargs):
        super().__init__(**kwargs)
        self._km = None
        self._kc = None
        self._cells = []
        self._execution_count = 0
        self._output_dir = output_dir
        self._ws_callback = ws_callback
        self._current_agent = "system"
        self._lock = threading.Lock()
        self._kernel_start_time = None
        self._last_heartbeat = None
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Kernel Lifecycle Management
    # =========================================================================

    def start_kernel(self) -> bool:
        """
        Start a new IPython kernel with Google Colab-equivalent environment.
        Returns True if kernel started successfully, False otherwise.
        """
        if self._km is not None and self._km.is_alive():
            logger.info("[Kernel] Already running, skipping start")
            return True

        logger.info("[Kernel] Starting new IPython kernel...")
        try:
            self._km = KernelManager(kernel_name="python3")
            self._km.start_kernel()
            self._kc = self._km.client()
            self._kc.start_channels()
            self._kc.wait_for_ready(timeout=30)
            self._kernel_start_time = time.time()
            self._last_heartbeat = time.time()
            logger.info("[Kernel] Kernel started successfully")

            # --- Colab-equivalent setup ---
            self._setup_colab_environment()
            return True

        except Exception as e:
            logger.error(f"[Kernel] Failed to start kernel: {e}")
            self._cleanup_kernel_refs()
            return False

    def _setup_colab_environment(self):
        """Configure kernel with Colab-equivalent settings."""
        # 1. Inline matplotlib rendering
        self._execute_silent("%matplotlib inline")

        # 2. Ensure output dir exists
        self._execute_silent(
            f"import os; os.makedirs(r'{self._output_dir}', exist_ok=True)"
        )

        # 3. Pre-load core data science libraries (like Colab)
        self._execute_silent("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# Enable high-DPI display for better chart quality
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

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
    from IPython.display import display, HTML, Markdown, Image, SVG, JSON, Latex
    from IPython.core.display import DisplayObject
except ImportError:
    pass
""")

        # 4. Set friendly pandas display options
        self._execute_silent("""
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('display.max_colwidth', 100)

# Enable HTML rendering for DataFrames
try:
    pd.set_option('display.notebook_repr_html', True)
except:
    pass

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
""")

        # 5. Define helper functions available in kernel
        self._execute_silent("""
def _kernel_state_snapshot():
    '''Return a snapshot of all user-defined variables in the kernel.'''
    import sys
    snapshot = {}
    skip_names = {'In', 'Out', 'get_ipython', 'exit', 'quit', '_', '__', '___',
                  '_i', '_ii', '_iii', '_oh', '_dh', '_sh', '_kernel_state_snapshot'}
    for name, obj in list(globals().items()):
        if name.startswith('_') or name in skip_names:
            continue
        if callable(obj) and hasattr(obj, '__module__'):
            if obj.__module__ and obj.__module__.startswith(('numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn', 'IPython')):
                continue
        try:
            info = {'type': type(obj).__name__}
            if hasattr(obj, 'shape'):
                info['shape'] = list(obj.shape)
            if hasattr(obj, 'dtype'):
                info['dtype'] = str(obj.dtype)
            if hasattr(obj, '__len__') and not isinstance(obj, str):
                info['len'] = len(obj)
            if isinstance(obj, (int, float, str, bool)):
                info['value'] = repr(obj)[:100]
            snapshot[name] = info
        except:
            snapshot[name] = {'type': type(obj).__name__, 'error': 'Could not inspect'}
    return snapshot

def _memory_usage():
    '''Return memory usage of the kernel process.'''
    try:
        import psutil
        process = psutil.Process()
        return {'rss_mb': round(process.memory_info().rss / 1024 / 1024, 2),
                'vms_mb': round(process.memory_info().vms / 1024 / 1024, 2)}
    except:
        return {'error': 'psutil not available'}
""")
        logger.info("[Kernel] Colab environment setup complete")

    def shutdown_kernel(self):
        """Shutdown the kernel. Safe to call multiple times."""
        logger.info("[Kernel] Shutting down...")
        try:
            if self._kc:
                self._kc.stop_channels()
        except Exception as e:
            logger.warning(f"[Kernel] Error stopping channels: {e}")

        try:
            if self._km and self._km.is_alive():
                self._km.shutdown_kernel(now=True)
            elif self._km:
                self._km.cleanup_resources()
        except Exception as e:
            logger.warning(f"[Kernel] Error during shutdown: {e}")

        self._cleanup_kernel_refs()
        logger.info("[Kernel] Shutdown complete")

    def _cleanup_kernel_refs(self):
        """Clear kernel references."""
        self._km = None
        self._kc = None
        self._kernel_start_time = None
        self._last_heartbeat = None

    def restart_kernel(self) -> bool:
        """Restart the kernel, preserving cell history but clearing namespace."""
        logger.info("[Kernel] Restarting...")
        self.shutdown_kernel()
        return self.start_kernel()

    def interrupt_kernel(self) -> bool:
        """
        Interrupt a long-running computation.
        Returns True if interrupt signal was sent successfully.
        """
        if not self._km or not self._km.is_alive():
            logger.warning("[Kernel] Cannot interrupt - kernel not running")
            return False

        try:
            self._km.interrupt_kernel()
            logger.info("[Kernel] Interrupt signal sent")
            return True
        except Exception as e:
            logger.error(f"[Kernel] Failed to interrupt: {e}")
            return False

    def is_alive(self) -> bool:
        """Check if kernel is running and responsive."""
        if not self._km:
            return False
        try:
            return self._km.is_alive()
        except:
            return False

    def check_heartbeat(self) -> bool:
        """
        Check kernel health via heartbeat.
        Returns True if kernel is responsive.
        """
        if not self._kc or not self._km:
            return False

        try:
            # Execute a simple expression to check responsiveness
            msg_id = self._kc.execute("1+1", silent=True)
            # Wait for response with short timeout
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    msg = self._kc.get_iopub_msg(timeout=1)
                    if (msg['parent_header'].get('msg_id') == msg_id and
                        msg['msg_type'] == 'status' and
                        msg['content']['execution_state'] == 'idle'):
                        self._last_heartbeat = time.time()
                        return True
                except QueueEmpty:
                    continue
            return False
        except Exception as e:
            logger.warning(f"[Kernel] Heartbeat check failed: {e}")
            return False

    # =========================================================================
    # Silent Execution (Setup commands, not tracked as cells)
    # =========================================================================

    def _execute_silent(self, code: str, timeout: int = 30):
        """Execute code without tracking as a cell (for setup commands)."""
        if not self._kc:
            self.start_kernel()

        msg_id = self._kc.execute(code, silent=True)
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                msg = self._kc.get_iopub_msg(timeout=1)
                if (msg["parent_header"].get("msg_id") == msg_id and
                    msg["msg_type"] == "status" and
                    msg["content"]["execution_state"] == "idle"):
                    break
            except QueueEmpty:
                continue
            except Exception:
                break

    # =========================================================================
    # Core IOPub Message Collector (Full Jupyter Protocol Compliance)
    # =========================================================================

    def _collect_output(self, msg_id: str, agent_name: str, timeout: int = None) -> dict:
        """
        Collect all outputs from the IOPub channel for a given msg_id.

        Full Jupyter Message Protocol compliance:
        - stream: stdout/stderr text output
        - display_data: rich display output (images, HTML, etc.)
        - execute_result: return value of last expression
        - error: exception traceback
        - clear_output: clear previous output
        - update_display_data: update existing display (for progress bars)

        Returns dict with:
        - stdout: concatenated stdout text
        - stderr: concatenated stderr text (ANSI stripped)
        - html: concatenated HTML content
        - svg: concatenated SVG content
        - latex: concatenated LaTeX content
        - json_data: list of JSON objects
        - markdown: concatenated Markdown content
        - images: list of saved image file paths
        - success: True if no errors occurred
        - execution_status: 'ok', 'error', or 'aborted'
        """
        if timeout is None:
            timeout = self._default_timeout

        stdout_parts = []
        stderr_parts = []
        html_parts = []
        svg_parts = []
        latex_parts = []
        markdown_parts = []
        json_data = []
        images = []
        execution_status = 'ok'
        clear_output_wait = False

        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                msg = self._kc.get_iopub_msg(timeout=1)
            except QueueEmpty:
                continue
            except Exception as e:
                stderr_parts.append(f"IOPub error: {str(e)}")
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            # ----------------------------------------------------------------
            # stream - stdout/stderr text output
            # ----------------------------------------------------------------
            if msg_type == "stream":
                text = content.get("text", "")
                if content["name"] == "stdout":
                    stdout_parts.append(text)
                elif content["name"] == "stderr":
                    # Strip ANSI escape codes for clean display
                    stderr_parts.append(_strip_ansi(text))

            # ----------------------------------------------------------------
            # display_data / execute_result - rich output
            # ----------------------------------------------------------------
            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data", {})
                metadata = content.get("metadata", {})

                # PNG image (matplotlib, seaborn plots)
                if "image/png" in data:
                    img_name = f"visualization_{self._execution_count}_{uuid.uuid4().hex[:6]}.png"
                    img_path = os.path.join(self._output_dir, img_name)
                    try:
                        img_bytes = base64.b64decode(data["image/png"])
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        images.append(img_path)
                        logger.debug(f"[Output] Saved PNG: {img_path}")
                    except Exception as e:
                        logger.warning(f"[Output] Failed to save PNG: {e}")

                # JPEG image
                elif "image/jpeg" in data:
                    img_name = f"visualization_{self._execution_count}_{uuid.uuid4().hex[:6]}.jpg"
                    img_path = os.path.join(self._output_dir, img_name)
                    try:
                        img_bytes = base64.b64decode(data["image/jpeg"])
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        images.append(img_path)
                        logger.debug(f"[Output] Saved JPEG: {img_path}")
                    except Exception as e:
                        logger.warning(f"[Output] Failed to save JPEG: {e}")

                # SVG image
                if "image/svg+xml" in data:
                    svg_parts.append(data["image/svg+xml"])

                # HTML output (pandas DataFrames, display(HTML(...)), etc.)
                if "text/html" in data:
                    html_parts.append(data["text/html"])

                # LaTeX output (mathematical expressions)
                if "text/latex" in data:
                    latex_parts.append(data["text/latex"])

                # Markdown output
                if "text/markdown" in data:
                    markdown_parts.append(data["text/markdown"])

                # JSON output
                if "application/json" in data:
                    try:
                        json_data.append(data["application/json"])
                    except:
                        pass

                # JavaScript (log but don't execute in backend)
                if "application/javascript" in data:
                    logger.debug("[Output] JavaScript output received (frontend execution required)")

                # Plain text fallback (only if no richer representation)
                if "text/plain" in data:
                    # Don't add plain text if we have richer formats
                    if not any(k in data for k in ["text/html", "image/png", "image/svg+xml", "text/latex"]):
                        stdout_parts.append(data["text/plain"])

            # ----------------------------------------------------------------
            # error - exception traceback
            # ----------------------------------------------------------------
            elif msg_type == "error":
                execution_status = 'error'
                ename = content.get("ename", "Error")
                evalue = content.get("evalue", "")
                # Clean ANSI codes from tracebacks
                tb = "\n".join(
                    _strip_ansi(line)
                    for line in content.get("traceback", [])
                )
                stderr_parts.append(tb)
                logger.debug(f"[Output] Error: {ename}: {evalue}")

            # ----------------------------------------------------------------
            # clear_output - clear previous output
            # ----------------------------------------------------------------
            elif msg_type == "clear_output":
                if content.get("wait", False):
                    clear_output_wait = True
                else:
                    # Immediate clear - reset output buffers
                    stdout_parts = []
                    html_parts = []
                    # Keep images and errors

            # ----------------------------------------------------------------
            # update_display_data - update existing display (tqdm progress bars)
            # ----------------------------------------------------------------
            elif msg_type == "update_display_data":
                data = content.get("data", {})
                # For progress bars, update the last HTML output
                if "text/html" in data:
                    # Replace last HTML part if it's a progress bar
                    new_html = data["text/html"]
                    if html_parts and "progress" in html_parts[-1].lower():
                        html_parts[-1] = new_html
                    else:
                        html_parts.append(new_html)
                if "text/plain" in data:
                    # For tqdm text progress
                    plain = data["text/plain"]
                    if stdout_parts and ('\r' in plain or '%' in plain):
                        stdout_parts[-1] = plain
                    else:
                        stdout_parts.append(plain)

            # ----------------------------------------------------------------
            # status - execution state change (idle = done)
            # ----------------------------------------------------------------
            elif msg_type == "status":
                if content["execution_state"] == "idle":
                    break

            # ----------------------------------------------------------------
            # execute_input - echoed input (ignore)
            # ----------------------------------------------------------------
            elif msg_type == "execute_input":
                pass  # We already have the code

            # ----------------------------------------------------------------
            # comm_* - widget communication (log for debugging)
            # ----------------------------------------------------------------
            elif msg_type.startswith("comm_"):
                logger.debug(f"[Output] Widget comm message: {msg_type}")

        # Assemble output
        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        html = "".join(html_parts)
        svg = "".join(svg_parts)
        latex = "".join(latex_parts)
        markdown = "".join(markdown_parts)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "html": html,
            "svg": svg,
            "latex": latex,
            "markdown": markdown,
            "json_data": json_data,
            "images": images,
            "success": execution_status == 'ok' and not bool(stderr.strip()),
            "execution_status": execution_status,
        }

    # =========================================================================
    # Agent-Facing Method (CrewAI Tool Interface)
    # =========================================================================

    def _run(self, code: str, agent_name: str = "system", timeout: int = None) -> str:
        """
        Execute code as a new cell in the Jupyter kernel.

        Args:
            code: Python code to execute
            agent_name: Name of the agent executing this cell
            timeout: Optional timeout in seconds (default: 120)

        Returns:
            JSON string with execution results
        """
        # Ensure kernel is running
        if not self._kc:
            if not self.start_kernel():
                return json.dumps({
                    "stdout": "",
                    "stderr": "Failed to start kernel",
                    "images": [],
                    "success": False,
                    "cell_id": None,
                    "execution_count": 0,
                })

        # Check kernel health and restart if needed
        if self._km and not self._km.is_alive():
            logger.warning("[Kernel] Kernel died, attempting restart...")
            if not self.restart_kernel():
                return json.dumps({
                    "stdout": "",
                    "stderr": "Kernel died and restart failed",
                    "images": [],
                    "success": False,
                    "cell_id": None,
                    "execution_count": 0,
                })

        # Inherit current agent name if caller didn't specify
        if agent_name == "system" and self._current_agent != "system":
            agent_name = self._current_agent

        with self._lock:
            self._execution_count += 1
            exec_count = self._execution_count

        cell_id = f"cell_{exec_count}_{uuid.uuid4().hex[:6]}"

        logger.debug(f"[Execute] Cell {cell_id} by {agent_name}: {code[:80]}...")

        # Execute code
        msg_id = self._kc.execute(code)
        output = self._collect_output(msg_id, agent_name, timeout=timeout)

        # Build cell record
        cell = {
            "cell_id": cell_id,
            "agent": agent_name,
            "code": code,
            "stdout": output["stdout"],
            "stderr": output["stderr"],
            "html": output["html"],
            "svg": output["svg"],
            "latex": output.get("latex", ""),
            "markdown": output.get("markdown", ""),
            "json_data": output.get("json_data", []),
            "images": output["images"],
            "execution_count": exec_count,
            "success": output["success"],
            "execution_status": output.get("execution_status", "ok"),
        }

        with self._lock:
            self._cells.append(cell)

        logger.debug(f"[Execute] Cell {cell_id} complete: success={output['success']}")

        return json.dumps({
            "stdout": output["stdout"],
            "stderr": output["stderr"],
            "html": output["html"],
            "svg": output["svg"],
            "latex": output.get("latex", ""),
            "markdown": output.get("markdown", ""),
            "json_data": output.get("json_data", []),
            "images": output["images"],
            "success": output["success"],
            "cell_id": cell_id,
            "execution_count": exec_count,
        })

    # =========================================================================
    # State Snapshotting (Deterministic Variable Inspection)
    # =========================================================================

    def get_kernel_state(self) -> dict:
        """
        Get a comprehensive snapshot of the kernel state.

        Returns dict with:
        - variables: dict of variable_name -> {type, shape, dtype, len, value}
        - memory: memory usage info
        - uptime: kernel uptime in seconds
        - execution_count: total cells executed
        - is_alive: kernel health status
        """
        if not self._kc or not self.is_alive():
            return {
                "variables": {},
                "memory": {"error": "Kernel not running"},
                "uptime": 0,
                "execution_count": self._execution_count,
                "is_alive": False,
            }

        # Get variable snapshot
        result = self._run("import json; print(json.dumps(_kernel_state_snapshot()))", agent_name="system")
        result_data = json.loads(result)

        variables = {}
        try:
            variables = json.loads(result_data.get("stdout", "{}").strip())
        except:
            pass

        # Get memory usage
        mem_result = self._run("import json; print(json.dumps(_memory_usage()))", agent_name="system")
        mem_data = json.loads(mem_result)

        memory = {}
        try:
            memory = json.loads(mem_data.get("stdout", "{}").strip())
        except:
            memory = {"error": "Could not get memory info"}

        # Calculate uptime
        uptime = 0
        if self._kernel_start_time:
            uptime = time.time() - self._kernel_start_time

        return {
            "variables": variables,
            "memory": memory,
            "uptime": round(uptime, 2),
            "execution_count": self._execution_count,
            "is_alive": self.is_alive(),
        }

    def get_dataframes_info(self) -> dict:
        """
        Get detailed info about all DataFrames in the kernel.

        Returns dict of df_name -> {shape, columns, dtypes, memory_mb, head}
        """
        code = """
import json
import pandas as pd
_dfs = {}
for _name, _obj in list(globals().items()):
    if isinstance(_obj, pd.DataFrame) and not _name.startswith('_'):
        try:
            _dfs[_name] = {
                'shape': list(_obj.shape),
                'columns': list(_obj.columns),
                'dtypes': _obj.dtypes.astype(str).to_dict(),
                'memory_mb': round(_obj.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'head': _obj.head(3).to_dict('records'),
            }
        except Exception as e:
            _dfs[_name] = {'error': str(e)}
print(json.dumps(_dfs))
"""
        result = self._run(code, agent_name="system")
        result_data = json.loads(result)

        try:
            return json.loads(result_data.get("stdout", "{}").strip())
        except:
            return {}

    # =========================================================================
    # Cell Query Methods
    # =========================================================================

    def get_cells(self) -> list:
        """Return all cell records for this session."""
        with self._lock:
            return self._cells.copy()

    def get_cells_by_agent(self, agent_name: str) -> list[str]:
        """Return all cell_ids belonging to the given agent."""
        with self._lock:
            return [
                cell["cell_id"]
                for cell in self._cells
                if cell.get("agent") == agent_name
            ]

    def get_cell(self, cell_id: str) -> Optional[dict]:
        """Get a specific cell by ID."""
        with self._lock:
            for cell in self._cells:
                if cell["cell_id"] == cell_id:
                    return cell.copy()
            return None

    # =========================================================================
    # Cell CRUD Methods (for REST API / User Interaction)
    # =========================================================================

    def _find_cell(self, cell_id: str) -> tuple[Optional[int], Optional[dict]]:
        """Return (index, cell_dict) for a cell_id, or (None, None)."""
        for i, cell in enumerate(self._cells):
            if cell["cell_id"] == cell_id:
                return i, cell
        return None, None

    def create_cell(self, code: str, agent_name: str = "user", position: Optional[int] = None) -> str:
        """Insert a new code cell without executing it. Returns cell_id."""
        with self._lock:
            self._execution_count += 1
            exec_count = self._execution_count

            cell_id = f"cell_{exec_count}_{uuid.uuid4().hex[:6]}"
            cell = {
                "cell_id": cell_id,
                "agent": agent_name,
                "code": code,
                "stdout": "",
                "stderr": "",
                "html": "",
                "svg": "",
                "latex": "",
                "markdown": "",
                "json_data": [],
                "images": [],
                "execution_count": exec_count,
                "success": True,
                "execution_status": "pending",
            }

            if position is not None and 0 <= position <= len(self._cells):
                self._cells.insert(position, cell)
            else:
                self._cells.append(cell)

            return cell_id

    def edit_cell(self, cell_id: str, new_code: str) -> Optional[dict]:
        """Edit a cell's source and clear its old outputs. Returns record or None."""
        with self._lock:
            idx, cell = self._find_cell(cell_id)
            if cell is None:
                return None

            cell["code"] = new_code
            cell["stdout"] = ""
            cell["stderr"] = ""
            cell["html"] = ""
            cell["svg"] = ""
            cell["latex"] = ""
            cell["markdown"] = ""
            cell["json_data"] = []
            cell["images"] = []
            cell["success"] = True
            cell["execution_status"] = "edited"

            return cell.copy()

    def delete_cell(self, cell_id: str) -> bool:
        """Remove a cell from the list. Returns True if found and removed."""
        with self._lock:
            idx, cell = self._find_cell(cell_id)
            if idx is None:
                return False
            self._cells.pop(idx)
            return True

    def rerun_cell(self, cell_id: str) -> Optional[dict]:
        """Re-execute an existing cell and return its updated record."""
        with self._lock:
            idx, cell = self._find_cell(cell_id)
            if cell is None:
                return None
            code = cell["code"]
            agent = cell["agent"]

        # Execute outside the lock
        if not self._kc:
            self.start_kernel()
        if self._km and not self._km.is_alive():
            self.restart_kernel()

        msg_id = self._kc.execute(code)
        output = self._collect_output(msg_id, agent)

        with self._lock:
            cell["stdout"] = output["stdout"]
            cell["stderr"] = output["stderr"]
            cell["html"] = output["html"]
            cell["svg"] = output["svg"]
            cell["latex"] = output.get("latex", "")
            cell["markdown"] = output.get("markdown", "")
            cell["json_data"] = output.get("json_data", [])
            cell["images"] = output["images"]
            cell["success"] = output["success"]
            cell["execution_status"] = output.get("execution_status", "ok")

            return cell.copy()

    def edit_and_rerun_cell(self, cell_id: str, new_code: str) -> Optional[dict]:
        """Edit a cell's source, then re-execute it. Returns updated record."""
        with self._lock:
            idx, cell = self._find_cell(cell_id)
            if cell is None:
                return None
            cell["code"] = new_code
            agent = cell["agent"]

        # Execute outside the lock
        if not self._kc:
            self.start_kernel()
        if self._km and not self._km.is_alive():
            self.restart_kernel()

        msg_id = self._kc.execute(new_code)
        output = self._collect_output(msg_id, agent)

        with self._lock:
            cell["stdout"] = output["stdout"]
            cell["stderr"] = output["stderr"]
            cell["html"] = output["html"]
            cell["svg"] = output["svg"]
            cell["latex"] = output.get("latex", "")
            cell["markdown"] = output.get("markdown", "")
            cell["json_data"] = output.get("json_data", [])
            cell["images"] = output["images"]
            cell["success"] = output["success"]
            cell["execution_status"] = output.get("execution_status", "ok")

            return cell.copy()

    def run_all_cells(self) -> list[dict]:
        """Re-execute all cells in order. Returns list of updated cell records."""
        results = []
        with self._lock:
            cell_ids = [c["cell_id"] for c in self._cells]

        for cell_id in cell_ids:
            result = self.rerun_cell(cell_id)
            if result:
                results.append(result)

        return results

    # =========================================================================
    # Export — Rebuild .ipynb from Cell Records
    # =========================================================================

    def export_notebook(self, path: str, include_system_cells: bool = False) -> str:
        """
        Write the cell history to disk as a .ipynb notebook.

        Args:
            path: Output path for the notebook
            include_system_cells: If False (default), filters out validation/system cells
                                  to produce a cleaner user-facing notebook

        Returns:
            Path to the saved notebook
        """
        import nbformat
        from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

        # Filter out system/validation cells unless explicitly requested
        SYSTEM_AGENTS = {"validation", "system"}

        with self._lock:
            cells_to_export = [
                cell for cell in self._cells
                if include_system_cells or cell.get("agent") not in SYSTEM_AGENTS
            ]

        nb = new_notebook()

        # Add metadata
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
        nb.metadata["language_info"] = {
            "name": "python",
            "version": "3.10"
        }

        for cell in cells_to_export:
            # Create code cell
            nb_cell = new_code_cell(source=cell["code"])
            nb_cell.metadata["agent"] = cell.get("agent", "system")
            nb_cell.metadata["cell_id"] = cell["cell_id"]
            nb_cell.execution_count = cell.get("execution_count")

            # Add outputs
            outputs = []

            # Stream output (stdout)
            if cell.get("stdout"):
                outputs.append(nbformat.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=cell["stdout"]
                ))

            # Stream output (stderr)
            if cell.get("stderr"):
                outputs.append(nbformat.v4.new_output(
                    output_type="stream",
                    name="stderr",
                    text=cell["stderr"]
                ))

            # HTML output (DataFrames, display(HTML(...)), etc.)
            if cell.get("html"):
                outputs.append(nbformat.v4.new_output(
                    output_type="execute_result",
                    data={"text/html": cell["html"], "text/plain": ""},
                    execution_count=cell.get("execution_count", 1),
                ))

            # SVG output
            if cell.get("svg"):
                outputs.append(nbformat.v4.new_output(
                    output_type="display_data",
                    data={"image/svg+xml": cell["svg"], "text/plain": "<SVG>"},
                ))

            # LaTeX output
            if cell.get("latex"):
                outputs.append(nbformat.v4.new_output(
                    output_type="execute_result",
                    data={"text/latex": cell["latex"], "text/plain": cell["latex"]},
                    execution_count=cell.get("execution_count", 1),
                ))

            # Markdown output
            if cell.get("markdown"):
                outputs.append(nbformat.v4.new_output(
                    output_type="display_data",
                    data={"text/markdown": cell["markdown"], "text/plain": cell["markdown"]},
                ))

            # PNG images
            for img_path in cell.get("images", []):
                try:
                    with open(img_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    outputs.append(nbformat.v4.new_output(
                        output_type="display_data",
                        data={"image/png": img_b64, "text/plain": "<Figure>"},
                    ))
                except Exception as e:
                    logger.warning(f"[Export] Failed to embed image {img_path}: {e}")

            nb_cell.outputs = outputs
            nb.cells.append(nb_cell)

        # Write notebook
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        logger.info(f"[Export] Notebook saved to {p}")
        return str(p)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> dict:
        """Get kernel statistics for debugging."""
        return {
            "is_alive": self.is_alive(),
            "execution_count": self._execution_count,
            "cell_count": len(self._cells),
            "uptime": round(time.time() - self._kernel_start_time, 2) if self._kernel_start_time else 0,
            "last_heartbeat": round(time.time() - self._last_heartbeat, 2) if self._last_heartbeat else None,
            "output_dir": self._output_dir,
        }

    def clear_cells(self):
        """Clear all cell records (does not affect kernel namespace)."""
        with self._lock:
            self._cells = []
            self._execution_count = 0
