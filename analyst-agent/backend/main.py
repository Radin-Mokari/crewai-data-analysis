import json
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

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


from typing import Optional

# --- Path A: Full Analysis (fresh orchestrator, kernel shutdown on complete) ---
@app.post("/analyze")
async def start_analysis(dataset_path: str, prompt: str, background_tasks: BackgroundTasks, session_id: Optional[str] = None):
    if not session_id:
        _remove_orchestrator(dataset_path)
        session_id = db.create_session(dataset_path, prompt)
    else:
        db.append_message(session_id, "user", prompt)
    
    background_tasks.add_task(run_analysis, session_id, dataset_path, prompt)
    return {"session_id": session_id, "status": "started"}


async def run_analysis(session_id, dataset_path, prompt):
    logger.info(f"[Analysis] Starting session {session_id} for {dataset_path}")
    logger.info(f"[Analysis] WebSocket connected: {ws_manager.is_connected}")

    session_data = db.get_session(session_id)
    messages = session_data.get("messages", []) if session_data else []

    orchestrator = _get_or_create_orchestrator(dataset_path)
    try:
        result = await orchestrator.run(dataset_path, prompt, messages)
        logger.info(f"[Analysis] Session {session_id} completed successfully")
        logger.info(f"[Analysis] Completed agents: {result.get('completed', [])}")
        logger.info(f"[Analysis] Charts created: {len(result.get('charts', []))}")

        # Final assistant message summarizing completion
        db.append_message(session_id, "assistant", "Analysis completed.")
        db.save_result(session_id, result)

        # Automatically save results to analysis_results/run_<timestamp>/
        bundle = orchestrator.save_results_bundle(str(ANALYSIS_RESULTS_DIR))
        await ws_manager.broadcast({
            "type": "results_saved",
            "content": bundle,
            "timestamp": datetime.now().isoformat(),
        })

        with _orch_lock:
            _last_completed[session_id] = orchestrator
        await ws_manager.broadcast({"type": "done", "content": session_id, "timestamp": ""})
        logger.info(f"[Analysis] Session {session_id} done event broadcast")
    except Exception as e:
        logger.error(f"[Analysis] Session {session_id} failed: {e}", exc_info=True)
        db.append_message(session_id, "assistant", f"An error occurred: {str(e)}")
        db.save_error(session_id, str(e))
        await ws_manager.broadcast({"type": "done", "content": f"error:{session_id}", "timestamp": ""})


# --- Path B: Direct Single Agent (shared orchestrator per dataset) ---
@app.post("/agent/run")
async def run_single_agent(
    dataset_path: str,
    agent_name: str,
    prompt: str,
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None
):
    valid = [
        "cleaning", "eda", "visualization", "statistics",
        "feature_engineering", "class_imbalance", "report",
    ]
    if agent_name not in valid:
        return {"error": f"Unknown agent. Must be one of: {valid}"}
    
    if not session_id:
        session_id = db.create_session(dataset_path, f"[{agent_name}] {prompt}")
    else:
        db.append_message(session_id, "user", f"[{agent_name}] {prompt}")

    background_tasks.add_task(
        run_single_agent_task, session_id, dataset_path, agent_name, prompt
    )
    return {"session_id": session_id, "agent": agent_name, "status": "started"}


async def run_single_agent_task(session_id, dataset_path, agent_name, prompt):
    logger.info(f"[SingleAgent] Starting {agent_name} for session {session_id}")
    logger.info(f"[SingleAgent] WebSocket connected: {ws_manager.is_connected}")

    session_data = db.get_session(session_id)
    messages = session_data.get("messages", []) if session_data else []

    orchestrator = _get_or_create_orchestrator(dataset_path)
    try:
        result = await orchestrator.run_single_specialist(dataset_path, agent_name, prompt, messages)
        logger.info(f"[SingleAgent] {agent_name} completed for session {session_id}")
        
        db.append_message(session_id, "assistant", f"Specialist {agent_name} completed.")
        db.save_result(session_id, result)

        # When report agent completes, save results to analysis_results/
        if agent_name == "report":
            bundle = orchestrator.save_results_bundle(str(ANALYSIS_RESULTS_DIR))
            await ws_manager.broadcast({
                "type": "results_saved",
                "content": bundle,
                "timestamp": datetime.now().isoformat(),
            })

        with _orch_lock:
            _last_completed[session_id] = orchestrator
        await ws_manager.broadcast({"type": "done", "content": session_id, "timestamp": ""})
        logger.info(f"[SingleAgent] Session {session_id} done event broadcast")
    except Exception as e:
        logger.error(f"[SingleAgent] {agent_name} failed for session {session_id}: {e}", exc_info=True)
        db.append_message(session_id, "assistant", f"An error occurred: {str(e)}")
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
    cell_type: str = "code"
    dataset_path: Optional[str] = None

class CellEditBody(BaseModel):
    code: str


def _find_active_tool(dataset_path: Optional[str] = None) -> JupyterSessionTool | None:
    """Return the JupyterSessionTool from any active orchestrator."""
    if dataset_path:
        orch = _get_or_create_orchestrator(dataset_path)
        tool = orch.tool
        if tool._km is None:
            tool.start_kernel()
        return tool

    with _orch_lock:
        for orch in _active_orchestrators.values():
            if orch.tool._km is not None:
                return orch.tool
        for orch in _last_completed.values():
            if orch.tool._km is not None:
                return orch.tool

    # Fallback to a default scratchpad kernel if nothing is running and no dataset_path was provided
    orch = _get_or_create_orchestrator("scratchpad_session")
    tool = orch.tool
    if tool._km is None:
        tool.start_kernel()
    return tool


@app.post("/kernel/execute")
async def kernel_execute(body: CellExecuteBody):
    tool = _find_active_tool(body.dataset_path)
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No dataset_path provided and no active kernel found."})
        
    result_json = tool._run(body.code, agent_name=body.agent_name, cell_type=body.cell_type)
    record = json.loads(result_json)
    await ws_manager.broadcast({
        "type": "cell_update",
        "content": record,
        "timestamp": datetime.now().isoformat(),
    })
    return record


@app.get("/kernel/cells")
async def kernel_get_cells(dataset_path: Optional[str] = None):
    tool = _find_active_tool(dataset_path)
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No dataset_path provided and no active kernel found."})
    return tool.get_cells()


@app.put("/kernel/cells/{cell_id}")
async def kernel_edit_cell(cell_id: str, body: CellEditBody, dataset_path: Optional[str] = None):
    tool = _find_active_tool(dataset_path)
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No dataset_path provided and no active kernel found."})
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
async def kernel_delete_cell(cell_id: str, dataset_path: Optional[str] = None):
    tool = _find_active_tool(dataset_path)
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No dataset_path provided and no active kernel found."})
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
async def kernel_rerun_cell(cell_id: str, dataset_path: Optional[str] = None):
    tool = _find_active_tool(dataset_path)
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No dataset_path provided and no active kernel found."})
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
async def kernel_edit_and_rerun(cell_id: str, body: CellEditBody, dataset_path: Optional[str] = None):
    tool = _find_active_tool(dataset_path)
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No dataset_path provided and no active kernel found."})
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


# ---------------------------------------------------------------------------
# Kernel State & Introspection Endpoints (Colab-parity)
# ---------------------------------------------------------------------------

@app.get("/kernel/state")
async def kernel_get_state():
    """
    Get comprehensive kernel state snapshot including:
    - All variables with types, shapes, and values
    - Memory usage
    - Kernel uptime and health
    """
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    return tool.get_kernel_state()


@app.get("/kernel/dataframes")
async def kernel_get_dataframes():
    """
    Get detailed info about all DataFrames in the kernel:
    - Shape, columns, dtypes
    - Memory usage
    - First 3 rows preview
    """
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    return tool.get_dataframes_info()


@app.get("/kernel/stats")
async def kernel_get_stats():
    """Get kernel statistics for debugging."""
    tool = _find_active_tool()
    if not tool:
        return {"is_alive": False, "message": "No active kernel"}
    return tool.get_stats()


@app.post("/kernel/interrupt")
async def kernel_interrupt():
    """Interrupt a long-running computation in the kernel."""
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    success = tool.interrupt_kernel()
    return {"status": "interrupted" if success else "failed", "success": success}


@app.post("/kernel/restart")
async def kernel_restart():
    """
    Restart the kernel, clearing the namespace but preserving cell history.
    Useful when the kernel gets into a bad state.
    """
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    success = tool.restart_kernel()
    await ws_manager.broadcast({
        "type": "progress",
        "content": "Kernel restarted" if success else "Kernel restart failed",
        "timestamp": datetime.now().isoformat(),
    })
    return {"status": "restarted" if success else "failed", "success": success}


@app.get("/kernel/heartbeat")
async def kernel_heartbeat():
    """Check if the kernel is responsive (heartbeat check)."""
    tool = _find_active_tool()
    if not tool:
        return {"alive": False, "message": "No active kernel"}
    alive = tool.check_heartbeat()
    return {"alive": alive, "stats": tool.get_stats()}


@app.post("/kernel/run-all")
async def kernel_run_all_cells():
    """Re-execute all cells in order."""
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})

    results = tool.run_all_cells()

    # Broadcast each cell update
    for cell in results:
        await ws_manager.broadcast({
            "type": "cell_update",
            "content": cell,
            "timestamp": datetime.now().isoformat(),
        })

    return {
        "status": "completed",
        "cells_executed": len(results),
        "all_successful": all(c.get("success", False) for c in results),
    }


@app.post("/kernel/clear-cells")
async def kernel_clear_cells():
    """Clear all cell records (does not affect kernel namespace)."""
    tool = _find_active_tool()
    if not tool:
        return JSONResponse(status_code=400, content={"error": "No active kernel."})
    tool.clear_cells()
    return {"status": "cleared"}


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
        ws_manager.disconnect(websocket)


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


# ---------------------------------------------------------------------------
# Debug endpoints for troubleshooting WebSocket and analysis issues
# ---------------------------------------------------------------------------

@app.get("/debug/ws")
async def debug_websocket():
    """Get WebSocket connection status and statistics."""
    return {
        "ws_stats": ws_manager.get_stats(),
        "active_orchestrators": len(_active_orchestrators),
        "completed_sessions": len(_last_completed),
    }


@app.get("/debug/state")
async def debug_state():
    """Get current state of all active orchestrators."""
    states = {}
    with _orch_lock:
        for dataset_path, orch in _active_orchestrators.items():
            states[dataset_path] = {
                "kernel_alive": orch.tool._km is not None and orch.tool._km.is_alive() if orch.tool._km else False,
                "cells_count": len(orch.tool.get_cells()),
                "completed_agents": orch.state.get("completed", []),
                "charts_count": len(orch.state.get("charts", [])),
            }
    return {
        "active_orchestrators": states,
        "ws_connected": ws_manager.is_connected,
    }


@app.post("/debug/test-ws")
async def debug_test_websocket():
    """Send a test event to verify WebSocket connectivity."""
    test_event = {
        "type": "progress",
        "content": f"[TEST] WebSocket test message at {datetime.now().isoformat()}",
        "timestamp": datetime.now().isoformat(),
    }
    await ws_manager.broadcast(test_event)
    return {
        "status": "sent",
        "ws_connected": ws_manager.is_connected,
        "event": test_event,
    }
