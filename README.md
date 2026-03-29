# AI Data Analyst — Multi-Agent Analysis Platform

A dynamic hierarchical multi-agent system for automated data analysis, powered by CrewAI, Google Gemini 2.5 Flash, and a persistent Jupyter execution kernel.

## Architecture

### Core Components

- **Manager Agent**: Custom ReAct loop that dynamically decides which specialists to run, evaluates their outputs, and can request improvements.
- **7 Specialist Agents**:
  - `cleaning` — Handle missing values, duplicates, outliers, type fixes
  - `eda` — Explore distributions, correlations, patterns
  - `visualization` — Create 3-5 insightful charts
  - `statistics` — Run hypothesis tests, normality checks, correlation significance
  - `feature_engineering` — Create polynomial/interaction features, encode categoricals, scale
  - `class_imbalance` — Detect and fix class imbalance via SMOTE/undersampling
  - `report` — Synthesize findings into markdown report (no code execution)
- **Jupyter Kernel (shared session)**: A persistent IPython kernel per analysis session via `jupyter_client.KernelManager`. Variables persist across cells within the same kernel session.

### Execution Paths

- **Path A** (`POST /analyze`): Manager-driven full analysis with ReAct loop
  - OBSERVE → THINK → DECIDE → DELEGATE → CAPTURE → EVALUATE cycle
  - Manager can use `DELEGATE`, `IMPROVE`, or `COMPLETE` actions
  - Max 10 iterations, max 2 redos per agent, 5 total redos
  - Enforces report completion before allowing COMPLETE
- **Path B** (`POST /agent/run`): Run a single specialist directly
  - Reuses kernel and state if already initialized
  - One automatic retry on failure

### Key Features

- **Context Injection**: Specialists receive prior findings + kernel state to prevent hallucination
- **Quality Gate**: Evaluates each specialist's output for errors, meaningful content, valid kernel state
- **Visualization Coverage Check**: Verifies charts cover correlations, distributions, and class imbalance
- **IMPROVE Action**: Manager can request additional work from specialists (max 2 improvements per agent)
- **Smart Skip Criteria**: Auto-skip cleaning if data is clean, skip class_imbalance if balanced

## Project Structure

```
crewai-data-analysis-1/
├── .env                        # API keys (GEMINI_API_KEY required)
├── .env.example                # Template for .env
├── analyst-agent/
│   ├── backend/
│   │   ├── main.py             # FastAPI app: analysis, sessions, WS, kernel CRUD
│   │   ├── config.py           # .env loading, Gemini LLM factory
│   │   ├── orchestrator.py     # Path A + Path B orchestration, quality gate
│   │   ├── agents.py           # 7 specialists + manager agent factory
│   │   ├── prompts.py          # All system prompts (CORE_MODE, specialist, manager)
│   │   ├── tools.py            # JupyterSessionTool (jupyter_client + cell tracking)
│   │   ├── websocket_manager.py # WebSocket broadcaster
│   │   ├── database.py         # SQLite session persistence
│   │   └── requirements.txt
│   └── frontend/               # Next.js 16 + React 19 + Tailwind CSS
│       ├── src/
│       │   ├── app/
│       │   │   ├── layout.tsx
│       │   │   ├── page.tsx    # Main 3-panel layout
│       │   │   └── globals.css
│       │   ├── components/
│       │   │   ├── SessionSidebar.jsx   # Session history browser
│       │   │   ├── ChatSection.jsx      # Prompt input, agent chips, report display
│       │   │   ├── KernelView.jsx       # Notebook-style cell rendering
│       │   │   ├── ProcessLogs.jsx      # Real-time agent thoughts & quality events
│       │   │   └── HydrationFix.tsx
│       │   ├── hooks/
│       │   │   └── useWebSocket.ts      # WebSocket with reconnection
│       │   └── stores/
│       │       └── sessionStore.ts      # Zustand state management
│       ├── next.config.ts
│       └── package.json
├── analysis_results/           # Output reports, notebooks, and charts
└── *.csv                       # Test datasets
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 1. Environment Variables

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your-key-here
```

### 2. Backend

```bash
cd analyst-agent/backend
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name python3
```

### 3. Frontend

```bash
cd analyst-agent/frontend
npm install
```

## Running

You need **two terminals**, both from the project root:

### Terminal 1 — Backend

```bash
cd analyst-agent
uvicorn backend.main:app --reload --port 8000
```

### Terminal 2 — Frontend

```bash
cd analyst-agent/frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

### CLI Testing (no frontend needed)

```bash
cd analyst-agent
python -m backend.cli ../flight.csv
```

## Test Datasets

Three CSV datasets are available in the project root for testing:

| Dataset | Description |
|---------|-------------|
| `California_Housing_Prices.csv` | Housing data with numeric features |
| `The Titanic dataset.csv` | Classic dataset with mixed types and missing values |
| `flight.csv` | Flight data for analysis |

## API Endpoints

### Analysis & Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a CSV file |
| POST | `/analyze` | Start full analysis (Path A — manager-driven) |
| POST | `/agent/run` | Run a single specialist (Path B) |
| GET | `/sessions` | List past sessions |
| GET | `/sessions/{id}` | Get session details with results |
| POST | `/save-results` | Bundle report markdown + charts into timestamped folder |
| WS | `/ws` | WebSocket for real-time streaming updates |

### Kernel Cell CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/kernel/execute` | Execute code as a new cell |
| GET | `/kernel/cells` | List all cells in the active kernel |
| PUT | `/kernel/cells/{cell_id}` | Edit a cell (clears outputs) |
| DELETE | `/kernel/cells/{cell_id}` | Delete a cell |
| POST | `/kernel/cells/{cell_id}/rerun` | Re-execute a cell |
| POST | `/kernel/cells/{cell_id}/edit-and-rerun` | Edit + re-execute (edit-in-place) |
| POST | `/kernel/export` | Export notebook (`.ipynb`) for the active kernel |
| POST | `/kernel/shutdown` | Shutdown active Path-B kernel for a dataset |

## Frontend Layout

The UI is a 3-column dark-themed layout:

```
┌────────────┬──────────────────────────────────────────────┐
│            │              Chat Section                    │
│  Session   │  (upload, prompt, agent chips,              │
│  Sidebar   │   progress, final report)                   │
│            ├──────────────────┬───────────────────────────┤
│            │  Kernel View     │   Process Logs            │
│            │  (notebook-style │   (agent thoughts,        │
│            │   cells: code,   │    quality events)        │
│            │   output, images)│                           │
└────────────┴──────────────────┴───────────────────────────┘
```

### WebSocket Events

| Event Type | Destination | Content |
|------------|-------------|---------|
| `agent_thought` | ProcessLogs | Manager/agent reasoning |
| `quality_event` | ProcessLogs | PASS/FAIL/FORCE-PASS status |
| `cell_update` | KernelView | New cell with code, stdout, stderr, images |
| `progress` | ChatSection | Current operation status |
| `done` | ChatSection | Analysis complete signal |
| `results_saved` | ChatSection | Path to saved results bundle |

## Manager Decision Flow

```
1. OBSERVE: Build snapshot (dataset profile, completed tasks, summaries)
2. THINK: Analyze what needs to be done
3. DECIDE: Output one of:
   - DELEGATE:agent_name | specific instructions
   - IMPROVE:agent_name | what's missing and improvements needed
   - COMPLETE (only after report is done)
4. EXECUTE: Run the specialist with context injection
5. EVALUATE: Quality gate checks output
6. Loop back to OBSERVE
```

### Skip Criteria

- `cleaning`: Skip if no missing values and duplicates < 1%
- `class_imbalance`: Skip if all potential targets are balanced (<70/30)
- `feature_engineering`: Skip unless user requests modeling
- `statistics`: Optional, can skip for quick analysis

## Output Structure

After each analysis, results are saved to:

```
analysis_results/run_<timestamp>/
├── charts/
│   ├── visualization_1_abc123.png
│   ├── visualization_2_def456.png
│   └── ...
├── analysis_report_<timestamp>.md
└── analysis_notebook_<timestamp>.ipynb
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Google Gemini 2.5 Flash |
| Agent Framework | CrewAI |
| Code Execution | jupyter_client (KernelManager) |
| Backend | FastAPI, Python 3.10+ |
| Frontend | Next.js 16, React 19, Tailwind CSS 4 |
| State Management | Zustand |
| Syntax Highlighting | Prism.js |
| Markdown Rendering | react-markdown |
| Database | SQLite |

## License

MIT
