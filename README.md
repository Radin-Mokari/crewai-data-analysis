# AI Data Analyst — Multi-Agent Analysis Platform

A dynamic hierarchical multi-agent system for automated data analysis, powered by CrewAI, Google Gemini, and a persistent Jupyter execution kernel.

## Architecture

- **Manager Agent**: Custom ReAct loop that dynamically decides which specialists to run.
- **7 Specialist Agents**: Cleaning, EDA, Visualization, Statistics, Feature Engineering, Class Imbalance, Report.
- **Jupyter Kernel (shared session)**: A persistent IPython kernel per analysis session.
  - Code execution is handled via `nbclient`/`nbformat` (with tracked outputs), while the kernel lifecycle uses `jupyter_client.KernelManager`.
  - Variables persist across cells within the same kernel session.
- **Two Execution Paths**:
  - **Path A**: `POST /analyze` (manager-driven full analysis)
  - **Path B**: `POST /agent/run` (run a single specialist)
- **Quality Gate + Context Injection**: Specialists receive the required prior findings + kernel state.

## Project Structure

```
crewai-data-analysis-1/
├── .env                        # API keys (GEMINI_API_KEY required)
├── .env.example                # Template for .env
├── analyst-agent/
│   ├── backend/
│   │   ├── main.py             # FastAPI app: analysis, sessions, WS, and kernel CRUD (/kernel/*)
│   │   ├── config.py           # .env loading, Gemini LLM factory
│   │   ├── orchestrator.py     # Path A + Path B orchestration
│   │   ├── agents.py           # 7 specialists + manager agent
│   │   ├── prompts.py          # All system prompts
│   │   ├── tools.py            # JupyterSessionTool (nbclient/nbformat + cell CRUD)
│   │   ├── websocket_manager.py
│   │   ├── database.py         # SQLite session persistence
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx
│       │   ├── components/
│       │   │   ├── SessionSidebar.jsx
│       │   │   ├── ChatSection.jsx         # includes the prompt island UI + agent chips
│       │   │   ├── KernelView.jsx          # renders kernel cells + output
│       │   │   └── ProcessLogs.jsx          # renders process logs drawer/panel
│       │   ├── hooks/
│       │   │   └── useWebSocket.js
│       │   └── stores/
│       │       └── sessionStore.js (Zustand)
│       ├── package.json
│       └── vite.config.js
├── California_Housing_Prices.csv
├── The Titanic dataset.csv
├── flight.csv
└── analysis_results/          # Output reports and charts
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 1. Environment variables

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

Open **http://localhost:5173** in your browser.

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

### Analysis / sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a CSV file |
| POST | `/analyze` | Start full analysis (Path A — manager-driven) |
| POST | `/agent/run` | Run a single specialist (Path B) |
| GET | `/sessions` | List past sessions |
| GET | `/sessions/{id}` | Get session details with results |
| WS | `/ws` | WebSocket for real-time streaming updates |
| POST | `/save-results` | Bundle report markdown + charts |

### Kernel cell CRUD (backend capability)

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

The current UI is a 3-column layout:

- **Left**: `SessionSidebar` (session list + “New session”)
- **Middle**: `ChatSection`
  - prompt island UI (agent chips + tools + prompt input)
  - streamed “thought” output + final report markdown
- **Right**: `KernelView` (cell rendering) + `ProcessLogs`
