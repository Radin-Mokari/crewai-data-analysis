import json
import threading
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.websocket_manager import WebSocketManager
from backend.orchestrator import AnalysisOrchestrator
from backend.database import SessionDB
from backend.tools import JupyterSessionTool
from backend.agents import create_specialists, create_manager
from backend.config import make_gemini_llm

app = FastAPI(title="AI Data Analyst")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_manager = WebSocketManager()
db = SessionDB()

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/files", StaticFiles(directory=str(RESULTS_DIR)), name="files")

ANALYSIS_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "analysis_results"
ANALYSIS_RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Orchestrator registry — keeps Path B orchestrators alive per dataset so
# sequential single-agent runs share the same Jupyter kernel and state.
# Completed orchestrators are kept in _last_completed so /save-results works.
# ---------------------------------------------------------------------------
_active_orchestrators: dict[str, AnalysisOrchestrator] = {}
_last_completed: dict[str, AnalysisOrchestrator] = {}
_orch_lock = threading.Lock()


def _make_orchestrator() -> AnalysisOrchestrator:
    tool = JupyterSessionTool(output_dir=str(RESULTS_DIR / "charts"))
    llm_medium = make_gemini_llm(640, 256)     # CLAUDE.md spec: compact code output
    llm_long = make_gemini_llm(4096, 0)        # report agent: large output, no thinking budget

    specialists = create_specialists(tool, llm_medium, llm_long)
    manager = create_manager(make_gemini_llm(640, 256))
    return AnalysisOrchestrator(
        tool=tool,
        specialists=specialists,
        manager=manager,
        llm=llm_medium,
        ws_manager=ws_manager,
    )


def _get_or_create_orchestrator(dataset_path: str) -> AnalysisOrchestrator:
    """Return an existing orchestrator for this dataset, or create a new one."""
    with _orch_lock:
        orch = _active_orchestrators.get(dataset_path)
        if orch is not None and orch.tool._km is not None:
            return orch
        orch = _make_orchestrator()
        _active_orchestrators[dataset_path] = orch
        return orch


def _remove_orchestrator(dataset_path: str):
    with _orch_lock:
        _active_orchestrators.pop(dataset_path, None)


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    upload_dir = RESULTS_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    return {"file_path": str(file_path), "filename": file.filename}


# --- Path A: Full Analysis (fresh orchestrator, kernel shutdown on complete) ---
@app.post("/analyze")
async def start_analysis(dataset_path: str, prompt: str, background_tasks: BackgroundTasks):
    _remove_orchestrator(dataset_path)
    session_id = db.create_session(dataset_path, prompt)
    background_tasks.add_task(run_analysis, session_id, dataset_path, prompt)
    return {"session_id": session_id, "status": "started"}


async def run_analysis(session_id, dataset_path, prompt):
    orchestrator = _make_orchestrator()
    try:
        result = await orchestrator.run(dataset_path, prompt)
        db.save_result(session_id, result)
        with _orch_lock:
            _last_completed[session_id] = orchestrator
        await ws_manager.broadcast({"type": "done", "content": session_id, "timestamp": ""})
    except Exception as e:
        db.save_error(session_id, str(e))
        await ws_manager.broadcast({"type": "done", "content": f"error:{session_id}", "timestamp": ""})


# --- Path B: Direct Single Agent (shared orchestrator per dataset) ---
@app.post("/agent/run")
async def run_single_agent(
    dataset_path: str,
    agent_name: str,
    prompt: str,
    background_tasks: BackgroundTasks,
):
    valid = [
        "cleaning", "eda", "visualization", "statistics",
        "feature_engineering", "class_imbalance", "report",
    ]
    if agent_name not in valid:
        return {"error": f"Unknown agent. Must be one of: {valid}"}
    session_id = db.create_session(dataset_path, f"[{agent_name}] {prompt}")
    background_tasks.add_task(
        run_single_agent_task, session_id, dataset_path, agent_name, prompt
    )
    return {"session_id": session_id, "agent": agent_name, "status": "started"}


async def run_single_agent_task(session_id, dataset_path, agent_name, prompt):
    orchestrator = _get_or_create_orchestrator(dataset_path)
    try:
        result = await orchestrator.run_single_specialist(dataset_path, agent_name, prompt)
        db.save_result(session_id, result)
        with _orch_lock:
            _last_completed[session_id] = orchestrator
        await ws_manager.broadcast({"type": "done", "content": session_id, "timestamp": ""})
    except Exception as e:
        db.save_error(session_id, str(e))
        await ws_manager.broadcast({"type": "done", "content": f"error:{session_id}", "timestamp": ""})


# --- Kernel cleanup for Path B sessions ---
@app.post("/kernel/shutdown")
async def shutdown_kernel(dataset_path: str):
    """Explicitly shut down a Path B kernel session."""
    with _orch_lock:
        orch = _active_orchestrators.pop(dataset_path, None)
    if orch:
        orch.tool.shutdown_kernel()
        return {"status": "shutdown"}
    return {"status": "no_active_kernel"}


@app.post("/save-results")
async def save_results(session_id: str):
    """
    Bundle the report markdown + all chart images from a completed session
    into analysis_results/run_<timestamp>/.
    """
    with _orch_lock:
        orchestrator = _last_completed.get(session_id)

    if not orchestrator:
        return JSONResponse(
            status_code=404,
            content={"error": "No completed analysis found for this session. Run an analysis first."},
        )

    try:
        bundle = orchestrator.save_results_bundle(str(ANALYSIS_RESULTS_DIR))
        return {
            "status": "saved",
            "run_dir": bundle["run_dir"],
            "report_path": bundle["report_path"],
            "charts_copied": bundle["charts_copied"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Kernel Cell CRUD Endpoints (full parity for users and agents)
# ---------------------------------------------------------------------------

class CellExecuteBody(BaseModel):
    code: str
    agent_name: str = "user"

class CellEditBody(BaseModel):
    code: str


def _find_active_tool() -> JupyterSessionTool | None:
    """Return the JupyterSessionTool from any active orchestrator."""
    with _orch_lock:
        for orch in _active_orchestrators.values():
            if orch.tool._km is not None:
                return orch.tool
        for orch in _last_completed.values():
            if orch.tool._km is not None:
                return orch.tool
    return None


@app.post("/kernel/execute")
async def kernel_execute(body: CellExecuteBody):
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    result_json = tool._run(body.code, agent_name=body.agent_name)
    record = json.loads(result_json)
    await ws_manager.broadcast({
        "type": "cell_update",
        "content": record,
        "timestamp": datetime.now().isoformat(),
    })
    return record


@app.get("/kernel/cells")
async def kernel_get_cells():
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    return tool.get_cells()


@app.put("/kernel/cells/{cell_id}")
async def kernel_edit_cell(cell_id: str, body: CellEditBody):
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    record = tool.edit_cell(cell_id, body.code)
    if record is None:
        return JSONResponse(status_code=404, content={"error": f"Cell {cell_id} not found."})
    await ws_manager.broadcast({
        "type": "cell_edited",
        "content": {"cell_id": cell_id, "code": body.code},
        "timestamp": datetime.now().isoformat(),
    })
    return record


@app.delete("/kernel/cells/{cell_id}")
async def kernel_delete_cell(cell_id: str):
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    removed = tool.delete_cell(cell_id)
    if not removed:
        return JSONResponse(status_code=404, content={"error": f"Cell {cell_id} not found."})
    await ws_manager.broadcast({
        "type": "cell_delete",
        "content": {"cell_id": cell_id},
        "timestamp": datetime.now().isoformat(),
    })
    return {"status": "deleted", "cell_id": cell_id}


@app.post("/kernel/cells/{cell_id}/rerun")
async def kernel_rerun_cell(cell_id: str):
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    record = tool.rerun_cell(cell_id)
    if record is None:
        return JSONResponse(status_code=404, content={"error": f"Cell {cell_id} not found."})
    await ws_manager.broadcast({
        "type": "cell_update",
        "content": record,
        "timestamp": datetime.now().isoformat(),
    })
    return record


@app.post("/kernel/cells/{cell_id}/edit-and-rerun")
async def kernel_edit_and_rerun(cell_id: str, body: CellEditBody):
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    record = tool.edit_and_rerun_cell(cell_id, body.code)
    if record is None:
        return JSONResponse(status_code=404, content={"error": f"Cell {cell_id} not found."})
    await ws_manager.broadcast({
        "type": "cell_update",
        "content": record,
        "timestamp": datetime.now().isoformat(),
    })
    return record


@app.post("/kernel/export")
async def kernel_export():
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = str(ANALYSIS_RESULTS_DIR / f"notebook_{timestamp}.ipynb")
    try:
        path = tool.export_notebook(export_path)
        return {"status": "exported", "path": path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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
    except Exception:
        ws_manager.disconnect()


@app.on_event("shutdown")
async def cleanup_kernels():
    """Shut down all active Jupyter kernels on server exit."""
    with _orch_lock:
        for dataset_path, orch in _active_orchestrators.items():
            try:
                orch.tool.shutdown_kernel()
            except Exception:
                pass
        _active_orchestrators.clear()
