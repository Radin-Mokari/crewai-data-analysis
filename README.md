# CrewAI Data Analysis Pipeline

A multi-agent sequential pipeline for automated data analysis using CrewAI and Google Gemini, with optional DeepSeek R1 context compression.

## Features

- **Multi-Agent Architecture**: 10 specialized agents for each analysis phase
- **Sequential Pipeline**: Data loading → Inspection → Cleaning → Transformation → EDA → Visualization → Statistical Analysis → Report Generation
- **Core Mode Prompting**: Agents use dynamic column detection (no hardcoded column names)
- **Codified Prompting**: Analysis agents output pseudocode plans before execution
- **Token Compression**: Optional DeepSeek R1 summarization between phases
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Automated Reporting**: Generates markdown reports with analysis findings

## Requirements

- Python 3.9+
- Google Gemini API key (required)
- OpenRouter API key (optional, for DeepSeek R1 compression)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd crewai-data-analysis
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

CrewAI has a SIGHUP bug on Windows. Run the fix script:

```bash
python fix_bug.py
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GEMINI_API_KEY=your-gemini-api-key-here
DATASET_PATH=Housing.csv
OUTPUT_DIR=./analysis_results

# Optional: For DeepSeek R1 context compression
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

**Get your API keys:**
- Gemini: https://aistudio.google.com/app/apikey
- OpenRouter (free): https://openrouter.ai (no credit card required)

## Usage

### Run the analysis pipeline

```bash
python run.py
```

The pipeline will:
1. Load and analyze your dataset
2. Clean and transform the data
3. Perform exploratory data analysis
4. Generate visualizations
5. Run statistical tests
6. Create a markdown report

### Output

Results are saved to timestamped directories:
```
analysis_results/
└── run_20251230_220626/
    ├── analysis_report_20251230_220626.md
    └── charts/
        ├── chart_1.png
        ├── chart_2.png
        └── chart_3.png
```

## Project Structure

```
├── crewai_data_analysis.py  # Main pipeline code with agents and tasks
├── run.py                   # Entry point script
├── fix_bug.py               # Windows SIGHUP bug fix
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── .gitignore               # Git ignore rules
├── Housing.csv              # Sample dataset
└── analysis_results/        # Generated reports and charts
```

## Agent Architecture

| Agent | Role | LLM Config |
|-------|------|------------|
| library_import | Environment verification | Short (320 tokens) |
| data_loading | Data structure summary | Short (320 tokens) |
| data_inspection | Quality inspection | Medium (640 tokens) |
| data_validation | Validation rules | Medium (640 tokens) |
| data_cleaning | Data cleaning | Medium (640 tokens) |
| data_transformation | Feature engineering | Medium (640 tokens) |
| eda_analysis | Exploratory analysis | Medium (640 tokens) |
| visualizations | Chart generation | Short (320 tokens) |
| statistical_tests | Statistical tests | Medium (640 tokens) |
| report_generator | Markdown report | Long (1200 tokens) |

## Prompting Strategies

- **Core Mode**: Agents reference pre-loaded metadata variables (`DATASET_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`) instead of hardcoding column names
- **Codified Prompting**: Analysis agents output structured pseudocode before execution
- **Inspector Pattern**: Agents self-correct errors by reading tracebacks and retrying
- **Token Stratification**: Report agent focuses on high-value insights, ignoring code blocks and verbose logs

## Token Compression (Optional)

When `OPENROUTER_API_KEY` is set, the pipeline uses DeepSeek R1 to summarize context between phases:

```
Phase 1 (6 agents) → [DeepSeek R1 Summary] → Phase 2 (3 agents) → [DeepSeek R1 Summary] → Report
```

This reduces token usage by ~80% for context passed between phases.

**Fallback**: If OpenRouter is unavailable, rule-based compression is used automatically.

## Troubleshooting

### Windows: SIGHUP Error
```
AttributeError: module 'signal' has no attribute 'SIGHUP'
```
**Fix:** Run `python fix_bug.py`

### Unicode Encoding Errors
```
'charmap' codec can't encode character
```
**Fix:** These are cosmetic logging issues and don't affect the analysis. To reduce them, run with UTF-8:
```powershell
chcp 65001
$env:PYTHONIOENCODING="utf-8"
python run.py
```

### Google GenAI Provider Not Available
```
ImportError: Google Gen AI native provider not available
```
**Fix:** Ensure `crewai[google-genai]` is installed (included in requirements.txt)

### LiteLLM Fallback Error
```
ImportError: Fallback to LiteLLM is not available
```
**Fix:** Ensure `litellm` is installed (included in requirements.txt)

## License

MIT
