# Product Requirements Document (PRD): Agentic Data Analysis Platform

## 1. Product Overview
The project is a vibe-coding platform (similar to Lovable AI or Emergent AI) tailored specifically for data analysis. It utilizes a hierarchical and dynamic multi-agent system to automate CSV data analysis and pre-processing by generating and executing Python code within a shared, Colab-like Jupyter kernel.

## 2. Target Audience
Data scientists, business analysts, and regular users who want to perform comprehensive exploratory data analysis (EDA), data cleaning, statistical analysis, and feature engineering through natural language prompts.

## 3. Core Capabilities & Architecture

### 3.1 Hierarchical Multi-Agent System
The system consists of a Manager Agent orchestrating 7 Specialist Agents:
1. **Cleaning Agent:** Handles missing values, duplicates, and data standardization.
2. **EDA (Exploratory Data Analysis) Agent:** Discovers patterns, shapes, and distributions.
3. **Statistics Agent:** Performs statistical testing and extracts numeric insights.
4. **Visualization Agent:** Generates plots and charts.
5. **Feature Engineer Agent:** Creates new predictive features from existing data.
6. **Class Imbalance Agent:** Detects and addresses target variable imbalances (e.g., SMOTE, undersampling).
7. **Reporter Agent:** Synthesizes the notebook context, outputs, and visualizations into a comprehensive Markdown report.

### 3.2 Dynamic Orchestration & ReAct Loop
- **Quality Gate:** The Manager agent examines the outputs of specialist agents for meaningfulness and correctness.
- **ReAct Loop Mechanism:** If a specialist agent produces insufficient or erroneous outputs (or if an error cascaded from an earlier step), the Manager re-delegates the task to the responsible agent with improved, targeted instructions.
- **Dynamic Skipping:** The Manager dynamically evaluates the dataset profile. For instance, if the EDA and Statistics agents confirm there are no missing values, the Manager skips the Data Cleaning step.

### 3.3 Execution Workflows
The workflow supports two operation modes:
- **Path A (Full Analysis):** The user provides a dataset and a high-level prompt. The Manager agent decides the complete execution plan—which specialists to trigger, in what order, and what to skip—culminating in a finalized markdown report.
- **Path B (Single Agent Run):** The user directly converses with the Manager to invoke a single specialist agent alongside a specific prompt.

### 3.4 Jupyter-based Execution Kernel
- **Shared Kernel:** All agents write their code to a unified Jupyter kernel, maintaining state (e.g., loaded DataFrames) sequentially.
- **Self-Correction (Edit-in-place):** After a quality gate failure or ReAct loop retry, agents edit their previously generated notebook cells rather than appending new conflicting cells.
- **Google Colab-like Frontend:** The UI provides full Jupyter functionality. Users can:
  - Create new code/text cells.
  - Read, write, type, and edit code manually.
  - Delete and re-run cells (both their own and those created by agents).
- **Dynamic Input Support:** The kernel natively supports dynamically loaded CSV files of any shape or context.

### 3.5 Manager Agent Chat Interface
- **Conversational UI:** The Manager agent functions as a traditional AI chatbot.
- **Agentic Steps & Tool Streaming:** The UI exposes the AI's internal thought process, decision-making, and tool utilization using collapsible streaming blocks (similar to ChatGPT or Claude AI processing indicators).
- **Session Memory:** The chatbot maintains a context window of the conversation for the currently active session. A heavy database context is not required as the project is open-source and session-scoped.

## 4. Required Outputs
1. **Markdown Report:** A synthesized, comprehensive analysis document.
2. **Visualizations:** Saved PNG/SVG/HTML charts generated during the workflow.
3. **Jupyter Notebook (.ipynb):** The complete code execution history, accessible, reproducible, and fully interactive via the web interface.
