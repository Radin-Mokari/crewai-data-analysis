"""
Local FastAPI server: one DataAnalysisWorkflow, one chat lock, same supervisor rules as CLI.

Run: uvicorn server:app --host 127.0.0.1 --port 8765
Requires: DATASET_PATH, GEMINI_API_KEY (see .env.example).
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

_workflow: Optional[Any] = None
_chat_lock = asyncio.Lock()


def _build_workflow() -> Any:
    from crewai_data_analysis import DataAnalysisWorkflow

    dataset_path = os.getenv("DATASET_PATH", "").strip()
    output_dir = os.getenv("OUTPUT_DIR", "./analysis_results")
    resume = (os.getenv("RESUME_RUN_DIR") or "").strip() or None
    if not dataset_path:
        raise RuntimeError("DATASET_PATH is required")
    if not Path(dataset_path).exists():
        raise RuntimeError(f"Dataset not found: {dataset_path}")
    return DataAnalysisWorkflow(
        dataset_path=dataset_path,
        output_dir=output_dir,
        resume_from=resume,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _workflow
    try:
        _workflow = _build_workflow()
    except Exception as e:
        print(f"[SERVER] Workflow not started: {e}")
        _workflow = None
    yield
    _workflow = None


app = FastAPI(title="CrewAI Data Analysis (local)", lifespan=lifespan)


@app.get("/health")
async def health():
    if _workflow is None:
        return {"status": "no_workflow", "hint": "Set DATASET_PATH to a valid CSV and restart, or POST /reset"}
    return {
        "status": "ok",
        "run_id": _workflow.run_id,
        "run_dir": str(_workflow.run_output_dir),
    }


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1)
    user_prompt: Optional[str] = None


def _run_chat_sync(message: str, user_prompt: str) -> Dict[str, Any]:
    from crewai_data_analysis import SupervisorLoopState, build_manager_system_instruction

    wf = _workflow
    assert wf is not None
    wf.ensure_dynamic_brief()
    if not wf._dynamic_bootstrapped:
        wf.bootstrap_dynamic_supervisor_session("", None, seed_initial_chat=False)
    wf.session_user_goal = message.strip()
    if not getattr(wf, "_http_iv_ready", False):
        wf._http_supervisor_state = SupervisorLoopState(specialist_count=len(wf.run_history_dynamic))
        wf._http_iv_ready = True
    wf._append_manager_chat_record("message", "user", message)
    specialists = wf.get_interactive_specialists()
    system_instruction = build_manager_system_instruction(wf.brief_dict)
    step_delay = float(os.getenv("DYNAMIC_STEP_DELAY_SECONDS", "2"))
    max_specialist_steps = int(os.getenv("DYNAMIC_MAX_STEPS", "18"))
    mgr_cap_raw = os.getenv("INTERACTIVE_MAX_MANAGER_TURNS", "").strip()
    max_manager_turns = int(mgr_cap_raw) if mgr_cap_raw.isdigit() else 0
    log: List[str] = []
    seg = wf._run_interactive_supervisor_segment(
        user_prompt,
        specialists,
        system_instruction,
        wf._http_supervisor_state,
        step_delay=step_delay,
        max_specialist_steps=max_specialist_steps,
        max_manager_turns=max_manager_turns,
        log=log,
    )
    return {"outcome": seg.outcome, "lines": log, "run_id": wf.run_id}


@app.post("/chat")
async def chat(body: ChatIn):
    async with _chat_lock:
        if _workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized — check DATASET_PATH")
        up = (body.user_prompt or os.getenv("USER_ANALYSIS_PROMPT") or body.message.strip() or "").strip()
        if not up:
            raise HTTPException(
                status_code=400,
                detail="Set user_prompt in JSON, USER_ANALYSIS_PROMPT, or a non-empty message",
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: _run_chat_sync(body.message.strip(), up),
        )


class PipelineIn(BaseModel):
    user_prompt: Optional[str] = None
    follow_ups: Optional[List[str]] = None


def _run_pipeline_sync(user_prompt: str, follow_ups: Optional[List[str]]) -> Dict[str, Any]:
    wf = _workflow
    assert wf is not None
    wf.ensure_dynamic_brief()
    wf.session_user_goal = (user_prompt or "").strip()
    wf.run_dynamic_team_pipeline(
        user_prompt=user_prompt,
        followup_messages=follow_ups,
        skip_terminal_reporter=True,
    )
    wf._http_iv_ready = False
    wf._http_supervisor_state = None
    wf._interactive_specialists_cache = None
    wf._save_kernel_snapshot_safe()
    return {"ok": True, "run_id": wf.run_id, "specialist_steps": len(wf.run_history_dynamic)}


@app.post("/pipeline")
async def run_pipeline(body: PipelineIn):
    async with _chat_lock:
        if _workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        up = (body.user_prompt or os.getenv("USER_ANALYSIS_PROMPT") or "").strip()
        if not up:
            raise HTTPException(
                status_code=400,
                detail="Set user_prompt in JSON or USER_ANALYSIS_PROMPT in the environment",
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: _run_pipeline_sync(up, body.follow_ups),
        )


def _report_sync() -> Dict[str, Any]:
    wf = _workflow
    assert wf is not None
    wf.ensure_dynamic_brief()
    specialists = wf.get_interactive_specialists()
    fallback = (os.getenv("USER_ANALYSIS_PROMPT") or "").strip() or "Analysis report"
    up = wf._effective_user_goal(fallback)
    wf._run_dynamic_terminal_reporter(up, specialists)
    wf._save_report_to_file()
    return {"ok": True, "run_dir": str(wf.run_output_dir)}


@app.post("/report")
async def report():
    async with _chat_lock:
        if _workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _report_sync)


class ResetIn(BaseModel):
    resume_run_dir: Optional[str] = None


@app.post("/reset")
async def reset(body: ResetIn):
    global _workflow
    async with _chat_lock:
        if body.resume_run_dir:
            os.environ["RESUME_RUN_DIR"] = body.resume_run_dir.strip()
        else:
            os.environ.pop("RESUME_RUN_DIR", None)
        try:
            _workflow = _build_workflow()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "run_id": _workflow.run_id, "run_dir": str(_workflow.run_output_dir)}


if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", "8765"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
