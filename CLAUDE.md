# CLAUDE.md — Multi-Agent Data Analysis Platform

> Complete build specification. Follow phase by phase. Do not deviate from the architecture.

---

## PROJECT CONTEXT

You are transforming an existing linear pipeline prototype into a dynamic hierarchical multi-agent system with a Jupyter-based execution kernel and a web frontend.

### What exists now (the prototype in the repo)
- Single Python script with a fixed 10-step sequential pipeline
- PythonSessionTool using exec() with shared globals dict as the kernel
- 10 CrewAI agents that always run in order: library_import → data_loading → data_inspection → data_validation → data_cleaning → data_transformation → eda_analysis → visualizations → statistical_tests → report_generator
- Gemini 2.5 Flash via CrewAI with token-optimized configs
- Simple retry mechanism (max 2 retries per task)
- Markdown report generation, charts saved to disk
- CLI-only execution — no frontend, no API, no persistence, no dynamic planning

### What you are building
- Manager Agent with a custom ReAct loop that dynamically decides which specialists to run and in what order
- 7 specialist agents (consolidated from 10) that the Manager delegates to on demand
- **Jupyter-based kernel** via jupyter_client — each agent writes and executes code in separate cells while sharing one persistent IPython kernel session. Replaces the old exec() approach.
- Context Injection Layer that prevents hallucination between specialists
- Quality Gate that evaluates each specialist's output and can trigger redos
- Two execution paths: Path A (Manager-driven full analysis) and Path B (user-triggered single agent)
- FastAPI backend with REST endpoints and WebSocket streaming
- React frontend (dark theme, 5 panels) with a notebook-style Kernel View showing per-agent cells
- SQLite for session persistence and file storage for charts/reports

---

## TARGET FILE STRUCTURE

```
analyst-agent/
├── backend/
│   ├── main.py                    # FastAPI app, all REST endpoints, WebSocket endpoint
│   ├── config.py                  # .env loading, make_gemini_llm() factory
│   ├── orchestrator.py            # AnalysisOrchestrator class with both execution paths
│   ├── agents.py                  # create_specialists() [7 agents] + create_manager()
│   ├── prompts.py                 # All system prompts for all agents + manager
│   ├── tools.py                   # JupyterSessionTool — jupyter_client based, replaces old exec() tool
│   ├── websocket_manager.py       # Simple single-connection WebSocket broadcaster
│   ├── database.py                # SessionDB class, SQLite sessions table
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SessionSidebar.jsx
│   │   │   ├── ChatSection.jsx
│   │   │   ├── KernelView.jsx     # Notebook-style: shows cells per agent with code + output + images
│   │   │   ├── ProcessLogs.jsx
│   │   │   └── PromptIsland.jsx
│   │   ├── hooks/
│   │   │   └── useWebSocket.js
│   │   └── stores/
│   │       └── sessionStore.js
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
└── README.md
```

---

## ARCHITECTURE RULES (HARD CONSTRAINTS)

### Jupyter Kernel Model
- One IPython kernel per analysis session, started via `jupyter_client.KernelManager`
- All agents share the SAME kernel session — variables persist across cells
- Each agent code execution creates a NEW cell in the kernel (tracked by cell_id, agent name, execution_count)
- The tool captures stdout, stderr, and display_data (images) from IOPub messages
- Charts rendered via matplotlib are captured as base64 PNG from display_data and saved to disk
- The kernel is shut down when the analysis completes or on error
- The Inspection Phase runs its code as the first cells in the kernel
- The State Validation Prefix runs as its own cell before each specialist's code

### Execution Paths

**Path A — Full Analysis:**
`POST /analyze` → `orchestrator.run()` → Start Jupyter kernel → Inspection (cells 1-2) → Manager ReAct Loop (OBSERVE → THINK → DECIDE → DELEGATE → CAPTURE → EVALUATE → loop back) → COMPLETE → Shutdown kernel

**Path B — Direct Single Agent:**
`POST /agent/run` → `orchestrator.run_single_specialist()` → Start kernel if not running → Inspection if not done (cells 1-2) → Context Injection → single specialist cell(s) → Quality Gate → done

Both paths share: kernel session, Inspection, Context Injection, all 7 specialists, JupyterSessionTool, Quality Gate, Session State, WebSocket events, and persistence.

### Agent Architecture
- **Manager Agent**: Does NOT execute code. Only reasons and outputs `DELEGATE:agent_name | instructions` or `COMPLETE`. No hardcoded sequence.
- **7 Specialist Agents**: cleaning, eda, visualization, statistics, feature_engineering, class_imbalance, report. All use JupyterSessionTool EXCEPT report (synthesises from task_summaries, no code).
- **Inspection Phase**: Deterministic Python function, NOT an LLM agent. Runs code cells in the Jupyter kernel to load CSV and profile dataset.

### Context Injection (anti-hallucination)
Every specialist execution goes through `_run_specialist()` which:
1. Concatenates text summaries from ALL previously completed specialists into the task description
2. Executes a validation cell in the Jupyter kernel that prints which DataFrames exist and their shapes
3. Builds enriched task: CORE MODE instructions + available kernel variables + prior findings + Manager's specific instructions

### Quality Gate
`_evaluate_quality()` checks after each specialist:
- No Python error strings in the cell output (Traceback, Exception, KeyError, TypeError, ValueError)
- Output is not empty/trivial (length > 20 chars)
- Kernel state is valid: executes `print(type(df_raw).__name__)` as a cell and checks for "DataFrame"
- Path A failure: increment redo_count, don't add to completed, Manager re-delegates (max 5 total redos)
- Path B failure: one automatic retry, then force-complete

### Session State (Dual-Layer)
```python
# Layer 1: Real Python variables in the Jupyter kernel namespace
# df_raw, df_clean, df_features, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, DATASET_SHAPE
# These CANNOT be hallucinated — NameError if accessed when missing
# Persist across ALL cells in the same kernel session

# Layer 2: Text summaries in orchestrator.state dict
self.state = {
    "profile": None,           # dict from inspect_dataset()
    "completed": [],           # specialist names that passed quality gate
    "task_summaries": {},      # specialist_name → summary string (max 500 chars)
    "charts": [],              # chart file paths
    "errors": [],              # error messages
    "redo_count": 0,           # total redos this session (max 5)
}
```

### WebSocket Events
All events are JSON: `{"type": str, "content": any, "timestamp": str}`. Types:
- `agent_thought` → routed to ProcessLogs panel
- `cell_update` → routed to KernelView panel. Content: `{"cell_id": str, "agent": str, "code": str, "stdout": str, "stderr": str, "images": list[str], "execution_count": int}`
- `quality_event` → routed to ProcessLogs panel
- `progress` → routed to ChatSection panel

### Frontend Layout
```
┌────────────┬──────────────────────────────────────────────┐
│            │              Chat Section                    │
│  Session   │  (progress, agent responses,                │
│  Sidebar   │   inline charts, final report)              │
│            ├──────────────────┬───────────────────────────┤
│            │  Kernel View     │   Process Logs            │
│            │  (notebook-style │   (thoughts + quality)    │
│            │   cells: agent,  │                           │
│            │   code, output,  │                           │
│            │   images)        │                           │
├────────────┴──────────────────┴───────────────────────────┤
│  Prompt Island                                            │
│  [Upload CSV]  [prompt text...]  [🔍 Analyze]            │
│  [🧹 Clean] [📊 EDA] [🎨 Viz] [📐 Stats]               │
│  [🔧 Feature Eng.] [⚖️ Imbalance] [📝 Report]           │
└───────────────────────────────────────────────────────────┘
```

Dark theme colors:
```
bg: #0d1117, surface: #161b22, hover: #1c2128
border: #30363d
text-primary: #e6edf3, text-secondary: #8b949e, text-muted: #484f58
accent-blue: #58a6ff, accent-green: #3fb950, accent-red: #f85149, accent-orange: #d29922
```

Tech stack: React 18 + Vite, Tailwind CSS, Zustand, native WebSocket. No Redux, no Socket.IO.

---

## PHASE 1: RESTRUCTURE BACKEND + JUPYTER KERNEL + MANAGER AGENT

### Step 1.1: Create `backend/config.py`

```python
import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

def make_gemini_llm(max_output_tokens: int, thinking_budget: int = 0):
    generation_config = {"max_output_tokens": max_output_tokens}
    if thinking_budget > 0:
        generation_config["thinking"] = {"budget_tokens": thinking_budget}
    return LLM(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        config=generation_config,
    )
```

### Step 1.2: Create `backend/tools.py` — JupyterSessionTool

This is the biggest change from the prototype. Replace the old exec()-based PythonSessionTool with a Jupyter kernel-backed tool.

```python
import os
import uuid
import base64
import json
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from jupyter_client import KernelManager

class CodeInput(BaseModel):
    code: str = Field(description="Python code to execute")

class JupyterSessionTool(BaseTool):
    """
    Executes Python code in a persistent Jupyter IPython kernel.
    Each call creates a new cell. All cells share the same kernel
    namespace — variables persist across calls.
    """
    name: str = "python_session"
    description: str = "Execute Python code in a shared Jupyter kernel session. Variables persist between calls."
    args_schema: type[BaseModel] = CodeInput

    # Instance state (not Pydantic fields)
    _km: KernelManager | None = None
    _kc: object | None = None  # KernelClient
    _cells: list = []
    _execution_count: int = 0
    _output_dir: str = "./results/charts"
    _ws_callback: object | None = None  # async callback for WebSocket

    def __init__(self, output_dir: str = "./results/charts", ws_callback=None, **kwargs):
        super().__init__(**kwargs)
        self._km = None
        self._kc = None
        self._cells = []
        self._execution_count = 0
        self._output_dir = output_dir
        self._ws_callback = ws_callback
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def start_kernel(self):
        """Start a new IPython kernel. Call once per analysis session."""
        if self._km is not None:
            return  # Already running
        self._km = KernelManager(kernel_name='python3')
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        # Pre-configure matplotlib for inline rendering
        self._execute_silent("%matplotlib inline")
        self._execute_silent("import matplotlib; matplotlib.use('agg')")
        self._execute_silent(f"import os; os.makedirs('{self._output_dir}', exist_ok=True)")

    def shutdown_kernel(self):
        """Shutdown the kernel. Call when analysis is complete."""
        if self._kc:
            self._kc.stop_channels()
        if self._km:
            self._km.shutdown_kernel(now=True)
        self._km = None
        self._kc = None

    def _execute_silent(self, code: str):
        """Execute code without tracking as a cell (for setup commands)."""
        if not self._kc:
            self.start_kernel()
        msg_id = self._kc.execute(code)
        # Wait for completion
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=30)
                if msg['parent_header'].get('msg_id') == msg_id and msg['msg_type'] == 'status':
                    if msg['content']['execution_state'] == 'idle':
                        break
            except Exception:
                break

    def _run(self, code: str, agent_name: str = "system") -> str:
        """
        Execute code as a new cell in the Jupyter kernel.
        Returns JSON string with: stdout, stderr, images, success, cell_id
        """
        if not self._kc:
            self.start_kernel()

        self._execution_count += 1
        cell_id = f"cell_{self._execution_count}_{uuid.uuid4().hex[:6]}"

        msg_id = self._kc.execute(code)

        stdout_parts = []
        stderr_parts = []
        images = []

        # Collect outputs from IOPub channel
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=60)
            except Exception:
                stderr_parts.append("Timeout: kernel did not respond within 60s")
                break

            if msg['parent_header'].get('msg_id') != msg_id:
                continue

            msg_type = msg['msg_type']
            content = msg['content']

            if msg_type == 'stream':
                if content['name'] == 'stdout':
                    stdout_parts.append(content['text'])
                elif content['name'] == 'stderr':
                    stderr_parts.append(content['text'])

            elif msg_type == 'display_data' or msg_type == 'execute_result':
                data = content.get('data', {})
                if 'image/png' in data:
                    # Save image to disk
                    img_name = f"{agent_name}_{self._execution_count}_{uuid.uuid4().hex[:6]}.png"
                    img_path = os.path.join(self._output_dir, img_name)
                    img_bytes = base64.b64decode(data['image/png'])
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)
                    images.append(img_path)
                if 'text/plain' in data:
                    stdout_parts.append(data['text/plain'])

            elif msg_type == 'error':
                stderr_parts.append('\n'.join(content.get('traceback', [])))

            elif msg_type == 'status' and content['execution_state'] == 'idle':
                break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)
        success = len(stderr) == 0

        # Build cell record
        cell = {
            "cell_id": cell_id,
            "agent": agent_name,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "images": images,
            "execution_count": self._execution_count,
            "success": success,
        }
        self._cells.append(cell)

        result = json.dumps({
            "stdout": stdout,
            "stderr": stderr,
            "images": images,
            "success": success,
            "cell_id": cell_id,
            "execution_count": self._execution_count,
        })

        return result

    def get_cells(self) -> list:
        """Return all cell records for this session."""
        return self._cells.copy()
```

**Key differences from the old PythonSessionTool:**
- Uses `jupyter_client.KernelManager` instead of `exec()` with globals dict
- Each code execution is a tracked cell with cell_id, agent name, execution_count
- Images are captured from IOPub `display_data` messages (base64 PNG) and saved to disk
- Kernel namespace is persistent — variables set in cell 1 are available in cell 50
- `start_kernel()` must be called once per session; `shutdown_kernel()` on completion
- `_execute_silent()` for setup commands that don't need tracking (matplotlib config, etc.)
- `agent_name` parameter on `_run()` tags each cell with which agent created it

### Step 1.3: Create `backend/agents.py`

```python
from crewai import Agent
from backend.tools import JupyterSessionTool
from backend.prompts import (
    CLEANING_PROMPT, EDA_PROMPT, VIZ_PROMPT,
    STATS_PROMPT, FEATURE_ENG_PROMPT, CLASS_IMBALANCE_PROMPT,
    REPORT_PROMPT, MANAGER_PROMPT
)

def create_specialists(tool: JupyterSessionTool, llm_medium, llm_long):
    return {
        "cleaning": Agent(
            role="Data Cleaning Specialist",
            goal="Clean and prepare the dataset for analysis.",
            backstory=CLEANING_PROMPT,
            llm=llm_medium, tools=[tool], verbose=True,
        ),
        "eda": Agent(
            role="EDA Specialist",
            goal="Explore data patterns, distributions, and correlations.",
            backstory=EDA_PROMPT,
            llm=llm_medium, tools=[tool], verbose=True,
        ),
        "visualization": Agent(
            role="Visualization Specialist",
            goal="Create insightful charts based on data characteristics.",
            backstory=VIZ_PROMPT,
            llm=llm_medium, tools=[tool], verbose=True,
        ),
        "statistics": Agent(
            role="Statistical Analysis Expert",
            goal="Run appropriate statistical tests.",
            backstory=STATS_PROMPT,
            llm=llm_medium, tools=[tool], verbose=True,
        ),
        "feature_engineering": Agent(
            role="Feature Engineering Specialist",
            goal="Create new features, transform existing ones, and prepare the dataset for modelling.",
            backstory=FEATURE_ENG_PROMPT,
            llm=llm_medium, tools=[tool], verbose=True,
        ),
        "class_imbalance": Agent(
            role="Class Imbalance Specialist",
            goal="Detect and address class imbalance in target variables using resampling, weighting, or synthetic generation.",
            backstory=CLASS_IMBALANCE_PROMPT,
            llm=llm_medium, tools=[tool], verbose=True,
        ),
        "report": Agent(
            role="Report Generator",
            goal="Synthesize findings into a markdown report.",
            backstory=REPORT_PROMPT,
            llm=llm_long, tools=[], verbose=True,
        ),
    }

def create_manager(llm):
    return Agent(
        role="Lead Data Analyst",
        goal="Orchestrate data analysis by delegating to the right specialist at the right time.",
        backstory=MANAGER_PROMPT,
        llm=llm, allow_delegation=True, verbose=True,
    )
```

### Step 1.4: Create `backend/prompts.py`

Create prompt constants. Port the existing CORE MODE prompts from the prototype for cleaning/eda/viz/stats/report. Add new prompts for feature_engineering and class_imbalance. Add the MANAGER_PROMPT.

Each specialist prompt MUST include:
```
CORE MODE: Use existing variables in the shared Jupyter kernel.
Do NOT call pd.read_csv() — data is already loaded in a previous cell.
Use the best available DataFrame: df_features > df_clean > df_raw
All variables from previous cells are available in the kernel namespace.
```

The FEATURE_ENG_PROMPT should instruct the agent to:
- Create polynomial and interaction features for highly correlated numeric pairs
- Encode categorical variables (one-hot for low cardinality, label encode for high)
- Scale numeric features if needed
- Store results in df_features
- Update NUMERIC_COLUMNS and CATEGORICAL_COLUMNS accordingly

The CLASS_IMBALANCE_PROMPT should instruct the agent to:
- First check if a categorical target column exists and is imbalanced (>70/30 split)
- If imbalanced: apply SMOTE, random undersampling, or compute class weights
- If NOT imbalanced or no clear target: report that and do nothing
- Never assume a target column — check the data first

### Step 1.5: Create `backend/orchestrator.py`

This is the core file. The Jupyter kernel is started at the beginning of an analysis and shared across all agent executions.

#### `inspect_dataset(tool, dataset_path) -> dict`
Deterministic function (no LLM). Runs two cells in the Jupyter kernel:

Cell 1 — Load data:
```python
import pandas as pd
import numpy as np
df_raw = pd.read_csv(r'{dataset_path}')
df_clean = None
df_features = None
DATASET_COLUMNS = list(df_raw.columns)
NUMERIC_COLUMNS = df_raw.select_dtypes(include=[np.number]).columns.tolist()
CATEGORICAL_COLUMNS = df_raw.select_dtypes(include=['object','category']).columns.tolist()
DATASET_SHAPE = df_raw.shape
ORIGINAL_NUMERIC_COLUMNS = NUMERIC_COLUMNS.copy()
ORIGINAL_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS.copy()
```

Cell 2 — Profile:
```python
import json
profile = {
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
print(json.dumps(profile))
```

Both cells use `tool._run(code, agent_name="inspection")`. Parse the stdout JSON from cell 2 to get the profile dict.

#### `class AnalysisOrchestrator`

**`__init__(self, tool, specialists, manager, llm, ws_manager=None)`**
Initialize with shared JupyterSessionTool, all 7 specialists dict, manager agent, llm, optional WebSocket manager. Create self.state dict.

**`async run(self, dataset_path, user_prompt) -> dict`** [PATH A]
1. **Start Jupyter kernel**: `self.tool.start_kernel()`
2. Call `inspect_dataset()` — runs cells 1-2 in kernel, store profile in state, emit progress
3. Enter while loop (max 10 iterations):
   - OBSERVE: `_build_snapshot(user_prompt)`, emit agent_thought
   - THINK + DECIDE: `_ask_manager(snapshot)`, emit agent_thought
   - Parse with `_parse_decision()`
   - If COMPLETE: break
   - If DELEGATE: `_run_specialist(name, instructions)` — creates new cell(s) in kernel
   - CAPTURE: store summary, emit cell_update with full cell data
   - EVALUATE: `_evaluate_quality()` — runs a validation cell in kernel
4. **Shutdown kernel**: `self.tool.shutdown_kernel()`
5. Return self.state

**`async run_single_specialist(self, dataset_path, agent_name, user_prompt) -> dict`** [PATH B]
1. If kernel not running → `self.tool.start_kernel()`
2. If `self.state["profile"]` is None → call `inspect_dataset()`
3. `_run_specialist(agent_name, user_prompt)` — creates cell(s) in kernel
4. CAPTURE + EVALUATE (one retry on failure)
5. Return self.state (do NOT shutdown kernel — user may run more agents in same session)

**`_build_snapshot(self, user_prompt) -> str`**
Build text snapshot for Manager: USER REQUEST, DATASET dims, COLUMNS, NUMERIC, CATEGORICAL, MISSING VALUES, DUPLICATES, COMPLETED TASKS, TASK SUMMARIES (each truncated to 200 chars), ERRORS (last 3).

**`_ask_manager(self, snapshot) -> str`**
Create a Task with the dynamic Manager prompt (see below). Execute via `Crew(agents=[self.manager], tasks=[task], process=Process.sequential)`.

Manager prompt must list all 7 specialists with descriptions:
```
AVAILABLE SPECIALISTS:
- cleaning: Fix missing values, duplicates, type errors, outliers
- eda: Explore distributions, correlations, patterns, groupby analysis
- visualization: Create charts (scatter, histogram, heatmap, box, bar)
- statistics: Run hypothesis tests, normality, correlation significance
- feature_engineering: Create new features, encode categoricals, scale numerics, polynomial/interaction features
- class_imbalance: Detect and fix class imbalance via SMOTE, undersampling, class weights, or stratified splitting
- report: Write a markdown summary of all findings (run this last)

DECISION GUIDELINES:
- Look at the USER REQUEST to understand what they actually want
- Look at COMPLETED TASKS to see what's already been done
- Look at the DATASET PROFILE to judge what's needed
- You can delegate specialists in ANY order, SKIP any, repeat any
- Only delegate ONE specialist at a time
- Say COMPLETE when the user's request has been fully addressed
- feature_engineering is typically useful after cleaning, before modelling
- class_imbalance is only needed if the dataset has a categorical target with skewed class distribution

RESPOND WITH EXACTLY ONE LINE:
DELEGATE:specialist_name | specific instructions for this specialist
or
COMPLETE
```

**`_parse_decision(self, decision) -> dict`**
Parse Manager output. Valid specialist names: cleaning, eda, visualization, statistics, feature_engineering, class_imbalance, report. Fallback to COMPLETE if unparseable.

**`_run_specialist(self, name, instructions) -> str`**
Context Injection Layer:
1. Get agent from `self.specialists[name]`
2. Build `prior_context` from `self.state["task_summaries"]` (each entry truncated to 300 chars)
3. Run State Validation cell in kernel via `self.tool._run(validation_code, agent_name="validation")`:
```python
_available = {}
for _name in ['df_raw', 'df_clean', 'df_features']:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, 'shape'):
        _available[_name] = list(_obj.shape)
print(f"Available DataFrames: {_available}")
print(f"NUMERIC_COLUMNS: {globals().get('NUMERIC_COLUMNS', 'NOT SET')}")
print(f"CATEGORICAL_COLUMNS: {globals().get('CATEGORICAL_COLUMNS', 'NOT SET')}")
```
4. Create Task with enriched description including CORE MODE, available variables, prior findings, instructions
5. Execute via Crew. The agent will call `self.tool._run(code, agent_name=name)` internally — creating tracked cells.
6. After execution, emit `cell_update` WebSocket event for each new cell the agent created. Get cells via `self.tool.get_cells()` and diff against previously known cells.

**`_evaluate_quality(self, specialist_name, result) -> bool`**
- Check 1: No error signal strings in result
- Check 2: `result.strip()` length > 20
- Check 3: Execute validation cell `print(type(df_raw).__name__)` in kernel, check "DataFrame" in stdout
- Return True only if all 3 pass

**`async _emit(self, event_type, content)`**
If self.ws exists, call `await self.ws.broadcast({"type": event_type, "content": content, "timestamp": datetime.now().isoformat()})`.

### Step 1.6: Verify via CLI

Create temporary `backend/cli.py`:
```python
import asyncio
from backend.orchestrator import AnalysisOrchestrator, inspect_dataset
from backend.tools import JupyterSessionTool
from backend.agents import create_specialists, create_manager
from backend.config import make_gemini_llm

async def main():
    tool = JupyterSessionTool(output_dir="./results/charts")
    llm_medium = make_gemini_llm(640, 256)
    llm_long = make_gemini_llm(1200, 512)
    specialists = create_specialists(tool, llm_medium, llm_long)
    manager = create_manager(make_gemini_llm(640, 256))
    orchestrator = AnalysisOrchestrator(
        tool=tool, specialists=specialists,
        manager=manager, llm=llm_medium
    )
    # Test Path A
    result = await orchestrator.run("sample_data.csv", "Analyze this dataset completely")
    print(f"Completed: {result['completed']}")
    print(f"Cells created: {len(tool.get_cells())}")
    for cell in tool.get_cells():
        print(f"  [{cell['agent']}] Cell {cell['execution_count']}: {cell['code'][:60]}...")

asyncio.run(main())
```

Verify: kernel starts, cells are created per agent, variables persist between cells, charts are captured, kernel shuts down cleanly.

---

## PHASE 2: FASTAPI BACKEND + WEBSOCKET

### Step 2.1: Create `backend/websocket_manager.py`

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional

class WebSocketManager:
    def __init__(self):
        self.active: Optional[WebSocket] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active = websocket

    def disconnect(self):
        self.active = None

    async def broadcast(self, event: dict):
        if self.active:
            try:
                await self.active.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                self.active = None
```

### Step 2.2: Create `backend/database.py`

```python
import sqlite3, json
from datetime import datetime
from pathlib import Path
import uuid

class SessionDB:
    def __init__(self, path="./results/sessions.db"):
        Path(path).parent.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                dataset_path TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                result_json TEXT,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )
        """)
        self.db.commit()

    def create_session(self, dataset_path, prompt):
        sid = str(uuid.uuid4())[:8]
        self.db.execute(
            "INSERT INTO sessions (id, dataset_path, prompt) VALUES (?,?,?)",
            (sid, dataset_path, prompt)
        )
        self.db.commit()
        return sid

    def save_result(self, session_id, result):
        self.db.execute(
            "UPDATE sessions SET status='complete', result_json=?, completed_at=? WHERE id=?",
            (json.dumps(result, default=str), datetime.now().isoformat(), session_id)
        )
        self.db.commit()

    def save_error(self, session_id, error):
        self.db.execute(
            "UPDATE sessions SET status='error', error=?, completed_at=? WHERE id=?",
            (error, datetime.now().isoformat(), session_id)
        )
        self.db.commit()

    def get_sessions(self, limit=20):
        rows = self.db.execute(
            "SELECT id, dataset_path, prompt, status, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row:
            result = dict(row)
            if result.get('result_json'):
                result['result'] = json.loads(result['result_json'])
            return result
        return None
```

### Step 2.3: Create `backend/main.py`

```python
from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.websocket_manager import WebSocketManager
from backend.orchestrator import AnalysisOrchestrator
from backend.database import SessionDB
from backend.tools import JupyterSessionTool
from backend.agents import create_specialists, create_manager
from backend.config import make_gemini_llm

app = FastAPI(title="AI Data Analyst")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ws_manager = WebSocketManager()
db = SessionDB()

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/files", StaticFiles(directory=str(RESULTS_DIR)), name="files")

def _make_orchestrator():
    tool = JupyterSessionTool(output_dir=str(RESULTS_DIR / "charts"))
    llm_medium = make_gemini_llm(640, 256)
    llm_long = make_gemini_llm(1200, 512)
    specialists = create_specialists(tool, llm_medium, llm_long)
    manager = create_manager(make_gemini_llm(640, 256))
    return AnalysisOrchestrator(
        tool=tool, specialists=specialists,
        manager=manager, llm=llm_medium, ws_manager=ws_manager
    )

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    upload_dir = RESULTS_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    return {"file_path": str(file_path), "filename": file.filename}

# --- Path A: Full Analysis ---
@app.post("/analyze")
async def start_analysis(dataset_path: str, prompt: str, background_tasks: BackgroundTasks):
    session_id = db.create_session(dataset_path, prompt)
    background_tasks.add_task(run_analysis, session_id, dataset_path, prompt)
    return {"session_id": session_id, "status": "started"}

async def run_analysis(session_id, dataset_path, prompt):
    orchestrator = _make_orchestrator()
    try:
        result = await orchestrator.run(dataset_path, prompt)
        db.save_result(session_id, result)
    except Exception as e:
        db.save_error(session_id, str(e))

# --- Path B: Direct Single Agent ---
@app.post("/agent/run")
async def run_single_agent(dataset_path: str, agent_name: str, prompt: str, background_tasks: BackgroundTasks):
    valid = ["cleaning", "eda", "visualization", "statistics", "feature_engineering", "class_imbalance", "report"]
    if agent_name not in valid:
        return {"error": f"Unknown agent. Must be one of: {valid}"}
    session_id = db.create_session(dataset_path, f"[{agent_name}] {prompt}")
    background_tasks.add_task(run_single_agent_task, session_id, dataset_path, agent_name, prompt)
    return {"session_id": session_id, "agent": agent_name, "status": "started"}

async def run_single_agent_task(session_id, dataset_path, agent_name, prompt):
    orchestrator = _make_orchestrator()
    try:
        result = await orchestrator.run_single_specialist(dataset_path, agent_name, prompt)
        db.save_result(session_id, result)
    except Exception as e:
        db.save_error(session_id, str(e))

@app.get("/sessions")
async def list_sessions():
    return db.get_sessions()

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    return db.get_session(session_id)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        ws_manager.disconnect()
```

### Step 2.4: Create `backend/requirements.txt`

```
fastapi>=0.110.0
uvicorn>=0.29.0
python-dotenv>=1.0.0
crewai>=0.120.0
python-multipart>=0.0.9
jupyter_client>=8.6.0
ipykernel>=6.29.0
```

---

## PHASE 3: REACT FRONTEND

### Step 3.1: Initialize project

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install zustand tailwindcss @tailwindcss/vite react-markdown prismjs
```

Configure Tailwind with the dark theme colors specified above.

### Step 3.2: Create `stores/sessionStore.js`

Zustand store with:
- `logs: []` — agent thoughts and quality events
- `cells: []` — Jupyter cells (cell_id, agent, code, stdout, stderr, images, execution_count)
- `progress: ""` — current progress message
- `sessions: []` — past sessions list
- `activeSession: null`
- `isRunning: false`
- Actions: `addLog(entry)`, `addCell(cell)`, `setProgress(msg)`, `setSessions(list)`, `setActiveSession(session)`, `setRunning(bool)`, `clearCurrent()` (resets logs, cells, progress)

### Step 3.3: Create `hooks/useWebSocket.js`

```javascript
import { useEffect, useRef } from 'react';
import { useSessionStore } from '../stores/sessionStore';

export function useWebSocket() {
  const ws = useRef(null);
  const addLog = useSessionStore(s => s.addLog);
  const addCell = useSessionStore(s => s.addCell);
  const setProgress = useSessionStore(s => s.setProgress);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'agent_thought':
        case 'quality_event':
          addLog(data);
          break;
        case 'cell_update':
          addCell(data.content);  // content has cell_id, agent, code, stdout, stderr, images
          break;
        case 'progress':
          setProgress(data.content);
          break;
      }
    };
    return () => ws.current?.close();
  }, []);

  return ws;
}
```

### Step 3.4: Create components

**App.jsx** — Grid layout matching the frontend layout spec. Dark theme. Initialize WebSocket hook.

**SessionSidebar.jsx** — `fetch('/sessions')` on mount → session list. Each item: dataset name, date, status badge (running=orange, complete=green, error=red). Click → load.

**ChatSection.jsx** — Scrolling column: user prompt, progress messages, inline chart images from `http://localhost:8000/files/charts/{filename}`, final markdown report via `react-markdown`.

**KernelView.jsx** — This is the notebook-style panel. Renders `cells[]` from store. Each cell shows:
- Agent name badge (color-coded by agent: cleaning=green, eda=blue, viz=purple, stats=orange, feature=teal, imbalance=pink, report=gray, inspection=indigo, validation=slate)
- Execution count number (like Jupyter's `In [3]:`)
- Python code block with syntax highlighting (Prism)
- Stdout output below the code
- Stderr output in red if present
- Inline images if the cell produced charts (render as `<img src="http://localhost:8000/files/charts/{filename}" />`)
- Auto-scroll to bottom when new cells arrive

**ProcessLogs.jsx** — Scrolling log of `logs[]`. Each entry: timestamp, agent name, text. Color: agent_thought=blue, quality pass=green, quality fail=red.

**PromptIsland.jsx** — Fixed bottom bar:
- File input for CSV upload (calls `POST /upload`, stores file_path)
- Text input for prompt
- "🔍 Analyze" button → `POST /analyze` [Path A]
- 7 agent buttons: 🧹 Clean, 📊 EDA, 🎨 Viz, 📐 Stats, 🔧 Feature Eng., ⚖️ Imbalance, 📝 Report → each calls `POST /agent/run` [Path B]
- Disabled until CSV uploaded, disabled while isRunning
- On click: `clearCurrent()` → `setRunning(true)` → API call

---

## PHASE 4: POLISH

### Step 4.1: Prompt Tuning
Iterate on `backend/prompts.py`. Key areas:
- Manager delegation logic
- Specialist CORE MODE instructions (must reference Jupyter kernel, not exec)
- Quality criteria

### Step 4.2: Error Handling
- try/catch around every Crew.kickoff()
- Gemini rate limit → wait 5s and retry once
- Specialist fails after 2 retries → skip (partial analysis > crash)
- **Always shutdown kernel** in a finally block to prevent zombie processes
- Frontend: show error state on WebSocket disconnect, "Analysis failed" on error status

### Step 4.3: Demo Datasets
3 datasets in `demo_data/`:
1. Clean dataset (e.g. Iris) — Manager skips cleaning
2. Messy dataset with nulls + outliers — full pipeline
3. Imbalanced classification dataset — class_imbalance agent activates

### Step 4.4: README
Setup instructions (including `pip install ipykernel` for Jupyter kernel), architecture overview, demo instructions.

---

## CRITICAL REMINDERS

1. **Jupyter kernel, not exec()**: Use `jupyter_client.KernelManager`. Each agent execution = new cell. Variables persist in kernel namespace across cells. Do NOT use the old exec() approach.
2. **Start/shutdown lifecycle**: `start_kernel()` at analysis start, `shutdown_kernel()` at completion or error. Use try/finally.
3. **Cell tracking**: Every `_run()` call creates a cell record with cell_id, agent, code, stdout, stderr, images, execution_count. Emit these via WebSocket for the KernelView.
4. **Image capture**: Matplotlib plots are captured from IOPub `display_data` messages as base64 PNG. Save to disk and include path in cell record.
5. **No hardcoded pipeline**: Manager decides order dynamically. NEVER force a sequence.
6. **Context injection is mandatory**: Every `_run_specialist()` MUST inject prior findings and run state validation cell.
7. **Two paths, shared kernel**: Path A and Path B use the SAME JupyterSessionTool, SAME kernel, SAME quality gate. Do not duplicate.
8. **Report agent has no tools**: Synthesises from task_summaries, does NOT execute code cells.
9. **7 agent buttons in frontend**: cleaning, eda, visualization, statistics, feature_engineering, class_imbalance, report.
10. **KernelView is notebook-style**: Show cells with execution count, agent badge, code, output, images — like a real Jupyter notebook.
