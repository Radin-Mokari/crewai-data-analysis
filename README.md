# CrewAI Data Analysis Pipeline

A multi-agent data analysis workflow using CrewAI and Google Gemini: **dynamic supervisor mode** (JSON routing over a shared Python kernel) or **sequential** legacy pipeline.

## Demo

<video width="100%" controls>
  <source src="docs/demo_compressed.MP4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Features

- **Multi-agent architecture**
  - **Dynamic mode** (default): Gemini **supervisor** + **7 specialists** (cleaning, feature engineering, class imbalance, EDA, visualization, statistics, reporter) plus routing outcomes **CHAT** / **DONE**.
  - **Sequential mode** (`WORKFLOW_MODE=sequential`): **10 phase agents** in fixed order (load → inspect → validate → clean → transform → EDA → visualize → statistics → report).
- **Persisted supervisor chat**: `manager_chat.jsonl` per run; long threads summarized when over `MANAGER_CHAT_BUDGET_CHARS` → optional `conversation_summary.txt`.
- **`chat_turn` rows**: Extra JSONL entries for CHAT replies (omitted from the manager prompt block to avoid duplicating assistant JSON).
- **Deterministic dataset brief**: `dataset_brief.txt` per run (time-series heuristics when applicable).
- **Kernel snapshots**: After each specialist step, optional `kernel_snapshot/` (Parquet + `meta.json`) for resume; `SESSION_SNAPSHOT=0` disables. Requires **pyarrow** (see `requirements.txt`).
- **Interactive supervisor (CLI)**: With **dynamic** mode, **`python run.py` keeps stdin open** so you can talk to the manager in the **same process/kernel** (`/report`, `exit` / `quit` / `q`).
- **Local HTTP API** (optional): `server.py` + Uvicorn — same rules as CLI; see [Usage](#usage).
- **Web UI** (optional): `chatbot-explorer-ui/` — Vite + React chat shell that calls the same API (`/chat`, `/report`, `/pipeline`, `/health`); see [Web UI (chatbot-explorer-ui)](#web-ui-chatbot-explorer-ui).
- **Core mode prompting**: Dynamic columns (`DATASET_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`); no hardcoded column names.
- **Codified prompting**: Analysis agents plan (pseudocode) before executing; inspector-style retries on errors.
- **Step delays & retries**: Configurable pause between supervisor turns (`DYNAMIC_STEP_DELAY_SECONDS`); task retries with backoff in code — not a separate "API rate limiter," but reduces burst requests.
- **Real-time Streaming Observability**: High-performance `asyncio` streaming pipeline for logs and Chain-of-Thought events. Uses packet padding and explicit flushing to ensure immediate UI updates.
- **Token Usage Tracing**: Precise real-time token tracking using **Tiktoken** (with resilient fallback). Token usage badges are displayed directly in the UI workflow logs.
- **Concurrency Stabilization**: Global execution locks to prevent background thread contention and GIL starvation during long-running agent workflows.

## Requirements

- Python 3.9+
- Google Gemini API key (`GEMINI_API_KEY`)
- **pyarrow** (for Parquet snapshots; installed via `requirements.txt`)
- AgentOps API key (optional, for monitoring)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <your-repo-folder>   # e.g. crewai-data-analysis-2
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Fix Windows-specific bug (if on Windows)

CrewAI can hit a SIGHUP issue on Windows. Run:

```bash
python fix_bug.py
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```
GEMINI_API_KEY=your-gemini-api-key-here
DATASET_PATH=your_dataset.csv
OUTPUT_DIR=./analysis_results

WORKFLOW_MODE=dynamic
USER_ANALYSIS_PROMPT=Your analysis goals in plain language
# Optional: resume — path to an existing run_* folder
# RESUME_RUN_DIR=./analysis_results/run_YYYYMMDD_HHMMSS

# Interactive after dynamic run: unset = on for dynamic; 0 = off; 1 = on
# INTERACTIVE_SESSION=

# Interactive-first (default 1): stdin sets goal before any batch. Set 0 for batch-from-env then stdin.
# INTERACTIVE_FIRST=1
# Batch-first: run full supervisor from USER_ANALYSIS_PROMPT before stdin (same as --batch-first)
# AUTO_RUN_SUPERVISOR=0
# After interactive-first, skip final generate_markdown_report unless /report filled content or set:
# AUTO_MARKDOWN_REPORT=0

# Optional: AgentOps monitoring
# AGENTOPS_API_KEY=your-agentops-api-key-here
```

**Get API keys:**

- Gemini: https://aistudio.google.com/app/apikey  
- AgentOps: https://app.agentops.ai (optional)

## Usage

### Run the analysis pipeline

```bash
python run.py
```

**Dynamic mode (default):** **Interactive-first** — opens the `>` prompt immediately; your first line is the analysis goal (no automatic batch from `USER_ANALYSIS_PROMPT`). For **batch-first** (full supervisor run from env), use `--batch-first` or set `AUTO_RUN_SUPERVISOR=1`.

CLI overrides (optional):

```bash
python run.py --workflow-mode dynamic --resume ./analysis_results/run_20260101_120000 --follow-up "Emphasize outliers in price"
python run.py --no-interactive
python run.py --interactive
python run.py --batch-first
```

**Interactive commands:** `/report` (regenerate full markdown report), `exit` / `quit` / `q`. `CHAT` supervisor turns do not consume `DYNAMIC_MAX_STEPS` (only specialist Crew runs do).

**Kernel snapshot (resume):** Outputs under `analysis_results/run_*/kernel_snapshot/`. Start with `RESUME_RUN_DIR` (or `--resume`) pointing at that folder to reload frames + chat/history.

**Local HTTP API:**

```bash
uvicorn server:app --host 127.0.0.1 --port 8765
```

- `GET /health` — liveness and `run_id`
- `GET /artifacts/{run_id}/charts/{filename}.png` — serve a chart from `OUTPUT_DIR/run_{run_id}/charts/` (path-safe; same folder the Python session writes to)
- `GET /artifacts/{run_id}/report` — raw `analysis_report_{run_id}.md` if present (`text/markdown`)
- `POST /pipeline` — one dynamic batch (`skip_terminal_reporter`); resets HTTP interactive state for a fresh chat session
- `POST /chat` — `{"message": "..."}`; optional `user_prompt` (falls back to `USER_ANALYSIS_PROMPT` or the message text). Response includes **`lines`** (manager log), plus **`specialist_steps`** (nested Crew outputs).
- `POST /report` — terminal-style reporter + save report; response includes **`report_markdown`** (file contents, capped by **`REPORT_MARKDOWN_MAX_CHARS`**, default 200000) and **`truncated`** when exceeded.
- `POST /reset` — rebuild workflow; optional `{"resume_run_dir": "..."}`

**CORS:** The server allows browser clients from common local Vite origins (port 8080) by default. Override with comma-separated **`CORS_ORIGINS`** in `.env` if you use another URL.

With **`WORKFLOW_MODE=dynamic`**, the supervisor chooses specialists using JSON decisions; transcripts go to `manager_chat.jsonl`. Use **`WORKFLOW_MODE=sequential`** for the fixed-order pipeline only.

### Web UI (chatbot-explorer-ui)

React + TypeScript app in **`chatbot-explorer-ui/`**. In development it proxies **`/api/*`** → `http://127.0.0.1:8765/*` (see `vite.config.ts`). The UI calls `GET /health`, `POST /chat`, `POST /report`, and `POST /pipeline`.

**Two terminals:**

1. **Backend** (repo root, venv active, `.env` with `DATASET_PATH` and `GEMINI_API_KEY`):

   ```bash
   uvicorn server:app --host 127.0.0.1 --port 8765
   ```

2. **Frontend:**

   ```bash
   cd chatbot-explorer-ui
   npm install
   npm run dev
   ```

Open the URL Vite prints (default **http://localhost:8080**). 

**Note on Memory:** If you encounter a "Process out of memory" error during `npm run dev`, the `package.json` includes an automatic fix that increases the Node.js memory limit to 4GB.

**Behavior notes:** The first user message in a sidebar session sets the **`user_prompt`** goal sent on every `/chat` until you start a new chat (client-side sessions only; the server keeps one workflow instance across all HTTP clients).

**Pipeline flow (summary):**

1. Load CSV into the shared Python session; write `dataset_brief.txt`.
2. **Dynamic:** supervisor loop → specialists + JSONL/history. **Sequential:** fixed agent chain.
3. Write `analysis_report_<run_id>.md` under the run folder (and interactive `/report` or post-loop generation as applicable).

### Output

Results go under timestamped directories, for example:

```
analysis_results/         # Tracked by default for persistence
└── run_20251230_220626/
    ├── analysis_report_20251230_220626.md
    ├── dataset_brief.txt
    ├── manager_chat.jsonl
    ├── run_history.json
    ├── session_meta.json
    ├── kernel_snapshot/           # Parquet + meta.json when snapshots enabled
    ├── conversation_summary.txt   # only if chat summarization ran (size limits)
    └── charts/
        ├── chart_<timestamp>_1.png
        └── ...
```

Chart files use **timestamped names**, not fixed `chart_1.png`.

## Project structure

```
├── crewai_data_analysis.py   # Agents, tasks, supervisor, workflow
├── session_state_store.py    # Kernel snapshot save/load
├── server.py                 # FastAPI app (optional)
├── run.py                    # CLI entry point
├── chatbot-explorer-ui/      # Vite + React UI (optional; npm install in that folder)
├── smoke_deterministic_brief.py
├── fix_bug.py                # Windows SIGHUP workaround
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/
│   └── demo_compressed.MP4   # Demo video
└── analysis_results/         # Created on run (gitignored if configured)
```

Sample CSVs (e.g. `Housing.csv`) may or may not be in the repo — set **`DATASET_PATH`** to your file.

## Agent architecture

### Sequential mode (`WORKFLOW_MODE=sequential`)

| Agent | Role | LLM (approx.) |
|-------|------|----------------|
| library_import | Environment verification | Short |
| data_loading | Data structure summary | Short |
| data_inspection | Quality inspection | Medium |
| data_validation | Validation rules | Medium |
| data_cleaning | Data cleaning | Medium |
| data_transformation | Feature engineering | Medium |
| eda_analysis | Exploratory analysis | Medium |
| visualizations | Chart generation | Short |
| statistical_tests | Statistical tests | Medium |
| report_generator | Markdown report | Long |

### Dynamic mode (`WORKFLOW_MODE=dynamic`)

Supervisor (Gemini JSON) routes to one of:

| Specialist key | Role |
|----------------|------|
| cleaning | Env check, validation, `df_clean` |
| feature_engineering | `df_features`, encoding/scaling |
| class_imbalance | Label distribution / imbalance |
| eda | Codified EDA |
| visualization | Charts (saved under `charts/`; headless Matplotlib `Agg` backend) |
| statistics | Statistical tests |
| reporter | Markdown synthesis from session facts + run digests |

Routing may also return **CHAT** (user-visible reply, no specialist Crew) or **DONE** (stop batch loop).

## Prompting strategies

- **Core mode**: Use `DATASET_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS` (and related globals) instead of hardcoding names.
- **Codified prompting**: Plan (pseudocode) then execute for analysis steps.
- **Inspector pattern**: Agents retry using tracebacks where applicable.
- **Reporter grounding**: Final reports are steered with **session facts** and **run-history excerpts** to reduce generic templates and bracket placeholders.

## AgentOps (optional)

When `AGENTOPS_API_KEY` is set, AgentOps traces agent/LLM activity. You should see a startup line confirming monitoring.

### Deterministic smoke test (no Gemini calls)

Loads a CSV and writes a brief (see script for args):

```bash
python smoke_deterministic_brief.py path/to/your.csv
```

## Troubleshooting

### Windows: SIGHUP error

```
AttributeError: module 'signal' has no attribute 'SIGHUP'
```

**Fix:** `python fix_bug.py`

### Unicode / console encoding

```
'charmap' codec can't encode character
```

**Fix (PowerShell):** `chcp 65001` and `$env:PYTHONIOENCODING="utf-8"` before `python run.py`.

### Google GenAI provider

```
ImportError: Google Gen AI native provider not available
```

**Fix:** `pip install -r requirements.txt` (includes `crewai[google-genai]`).

### Matplotlib: `FigureCanvasAgg is non-interactive`

Agent code may call `plt.show()` while the backend is **Agg** (no GUI). Figures should still be saved under `charts/` via `savefig` / executor behavior; the message is a **warning**, not a fatal error.

### Parquet / snapshot errors
Install **pyarrow**: `pip install pyarrow` (listed in `requirements.txt`).

### Vite: "Fatal process out of memory"
If the frontend crashes during startup, ensure you are running the project's custom `dev` script. The memory limit is already increased to 4GB in `package.json`:
```bash
npm run dev
```

### Server: No token counts showing
If token counts appear as "0", ensure `tiktoken` is installed in your active virtual environment:
```bash
pip install tiktoken
```
The system uses a resilient import, so it will not crash if the library is missing, but counts will be disabled.
