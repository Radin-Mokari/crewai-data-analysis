# CrewAI Data Analysis Pipeline

A multi-agent sequential pipeline for automated data analysis using CrewAI and Google Gemini.

## Features

- **Multi-Agent Architecture**: Specialized agents for each analysis phase
- **Sequential Pipeline**: Data loading → Inspection → Cleaning → Transformation → EDA → Visualization → Statistical Analysis → Report Generation
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Automated Reporting**: Generates markdown reports with analysis findings

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/crewai-data-analysis.git
   cd crewai-data-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

## Usage

Run the analysis pipeline:

```bash
python run.py
```

The pipeline will:
1. Load and analyze your dataset
2. Clean and transform the data
3. Perform exploratory data analysis
4. Generate visualizations
5. Run statistical tests
6. Create a markdown report in `analysis_results/`

## Project Structure

```
├── crewai_data_analysis.py  # Main pipeline code
├── run.py                   # Entry point script
├── requirements.txt         # Python dependencies
├── sample_data.csv          # Sample dataset
├── .env.example             # Template for environment variables
├── .gitignore               # Git ignore rules
└── analysis_results/        # Generated reports and charts
    ├── analysis_report.md
    └── charts/
```

## Requirements

- Python 3.9+
- Google Gemini API key
