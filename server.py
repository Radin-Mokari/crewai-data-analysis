"""
Local FastAPI server: one DataAnalysisWorkflow, one chat lock, same supervisor rules as CLI.

Run: uvicorn server:app --host 127.0.0.1 --port 8765
Requires: DATASET_PATH, GEMINI_API_KEY (see .env.example).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

load_dotenv()


def _cors_allow_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
    ]

_workflow: Optional[Any] = None
_chat_lock = asyncio.Lock()

# Max chars for specialist excerpts in POST /chat JSON; report body in POST /report.
_CHAT_EXCERPT_MAX = int(os.getenv("CHAT_EXCERPT_MAX_CHARS", "4000"))
_REPORT_MARKDOWN_MAX = int(os.getenv("REPORT_MARKDOWN_MAX_CHARS", "200000"))

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _output_dir_resolved() -> Path:
    return Path(os.getenv("OUTPUT_DIR", "./analysis_results")).resolve()


def _safe_run_id(run_id: str) -> str:
    rid = (run_id or "").strip()
    if not rid or not _RUN_ID_RE.match(rid):
        raise HTTPException(status_code=400, detail="Invalid run_id")
    return rid


def _run_artifacts_root(run_id: str) -> Path:
    rid = _safe_run_id(run_id)
    return _output_dir_resolved() / f"run_{rid}"


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    if _workflow is None:
        return {"status": "no_workflow", "hint": "Set DATASET_PATH to a valid CSV and restart, or POST /reset"}
    return {
        "status": "ok",
        "run_id": _workflow.run_id,
        "run_dir": str(_workflow.run_output_dir),
    }


@app.get("/artifacts/{run_id}/charts/{filename}")
async def artifact_chart(run_id: str, filename: str):
    """Serve a PNG from analysis_results/run_{run_id}/charts/ (path-safe)."""
    if not filename or filename != Path(filename).name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    lower = filename.lower()
    if not lower.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only .png charts are supported")
    root = _run_artifacts_root(run_id)
    path = (root / "charts" / filename).resolve()
    charts_root = (root / "charts").resolve()
    try:
        path.relative_to(charts_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(path, media_type="image/png")


@app.get("/artifacts/{run_id}/report", response_class=PlainTextResponse)
async def artifact_report_md(run_id: str):
    """Return analysis_report_{run_id}.md if present (UTF-8)."""
    root = _run_artifacts_root(run_id)
    path = root / f"analysis_report_{_safe_run_id(run_id)}.md"
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return PlainTextResponse(content=path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1)
    user_prompt: Optional[str] = None


def _sse_data_line(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _run_chat_sync(
    message: str,
    user_prompt: str,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
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
    n_hist = len(wf.run_history_dynamic)
    charts_dir = wf.run_output_dir / "charts"
    charts_before: Set[str] = set()
    if charts_dir.is_dir():
        charts_before = {p.name for p in charts_dir.glob("*.png")}
    seg = wf._run_interactive_supervisor_segment(
        user_prompt,
        specialists,
        system_instruction,
        wf._http_supervisor_state,
        step_delay=step_delay,
        max_specialist_steps=max_specialist_steps,
        max_manager_turns=max_manager_turns,
        log=log,
        emit=emit,
    )
    new_entries = wf.run_history_dynamic[n_hist:]
    specialist_steps: List[Dict[str, str]] = []
    for e in new_entries:
        excerpt = str(e.get("output_excerpt") or "")[:_CHAT_EXCERPT_MAX]
        specialist_steps.append(
            {
                "agent": str(e.get("agent") or ""),
                "excerpt": excerpt,
            }
        )
    chart_urls: List[str] = []
    if charts_dir.is_dir():
        for p in sorted(charts_dir.glob("*.png"), key=lambda x: x.stat().st_mtime_ns):
            if p.name not in charts_before:
                chart_urls.append(f"/artifacts/{wf.run_id}/charts/{p.name}")

    return {
        "outcome": seg.outcome,
        "lines": log,
        "run_id": wf.run_id,
        "specialist_steps": specialist_steps,
        "chart_urls": chart_urls,
        "manager_reply": seg.manager_reply or "",
    }


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


@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    """Server-Sent Events: supervisor events, then `final` with same JSON shape as POST /chat."""
    async with _chat_lock:
        if _workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized — check DATASET_PATH")
        up = (body.user_prompt or os.getenv("USER_ANALYSIS_PROMPT") or body.message.strip() or "").strip()
        if not up:
            raise HTTPException(
                status_code=400,
                detail="Set user_prompt in JSON, USER_ANALYSIS_PROMPT, or a non-empty message",
            )
        q: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue()

        def work() -> None:
            try:
                result = _run_chat_sync(body.message.strip(), up, emit=q.put)
                q.put({"type": "final", **result})
            except Exception as e:
                q.put({"type": "error", "message": str(e)})
            finally:
                q.put(None)

        async def event_gen():
            asyncio.get_running_loop().run_in_executor(None, work)
            while True:
                item = await asyncio.to_thread(q.get)
                if item is None:
                    break
                yield _sse_data_line(item)

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


class PipelineIn(BaseModel):
    user_prompt: Optional[str] = None
    follow_ups: Optional[List[str]] = None


def _run_pipeline_sync(
    user_prompt: str,
    follow_ups: Optional[List[str]],
    *,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    log: Optional[List[str]] = None,
) -> Dict[str, Any]:
    wf = _workflow
    assert wf is not None
    wf.ensure_dynamic_brief()
    wf.session_user_goal = (user_prompt or "").strip()
    wf.run_dynamic_team_pipeline(
        user_prompt=user_prompt,
        followup_messages=follow_ups,
        skip_terminal_reporter=True,
        emit=emit,
        log=log,
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


@app.post("/pipeline/stream")
async def run_pipeline_stream(body: PipelineIn):
    """SSE: supervisor events for full batch, then `final` with ok, run_id, specialist_steps, lines."""
    async with _chat_lock:
        if _workflow is None:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        up = (body.user_prompt or os.getenv("USER_ANALYSIS_PROMPT") or "").strip()
        if not up:
            raise HTTPException(
                status_code=400,
                detail="Set user_prompt in JSON or USER_ANALYSIS_PROMPT in the environment",
            )
        q: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue()
        log_lines: List[str] = []

        def work() -> None:
            try:
                result = _run_pipeline_sync(up, body.follow_ups, emit=q.put, log=log_lines)
                q.put(
                    {
                        "type": "final",
                        "ok": result["ok"],
                        "run_id": result["run_id"],
                        "specialist_steps": result["specialist_steps"],
                        "lines": list(log_lines),
                    }
                )
            except Exception as e:
                q.put({"type": "error", "message": str(e)})
            finally:
                q.put(None)

        async def event_gen():
            asyncio.get_running_loop().run_in_executor(None, work)
            while True:
                item = await asyncio.to_thread(q.get)
                if item is None:
                    break
                yield _sse_data_line(item)

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
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
    report_path = wf.report_path
    if report_path is None:
        report_path = wf.run_output_dir / f"analysis_report_{wf.run_id}.md"
    report_markdown = ""
    truncated = False
    if report_path.is_file():
        raw = report_path.read_text(encoding="utf-8")
        if len(raw) > _REPORT_MARKDOWN_MAX:
            report_markdown = raw[:_REPORT_MARKDOWN_MAX]
            truncated = True
        else:
            report_markdown = raw
    return {
        "ok": True,
        "run_dir": str(wf.run_output_dir),
        "report_markdown": report_markdown,
        "truncated": truncated,
    }


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
