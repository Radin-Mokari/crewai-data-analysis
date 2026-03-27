from crewai import Agent
from backend.tools import JupyterSessionTool
from backend.prompts import (
    CLEANING_PROMPT,
    EDA_PROMPT,
    VIZ_PROMPT,
    STATS_PROMPT,
    FEATURE_ENG_PROMPT,
    CLASS_IMBALANCE_PROMPT,
    REPORT_PROMPT,
    MANAGER_PROMPT,
)


def create_specialists(tool: JupyterSessionTool, llm_medium, llm_long):
    return {
        "cleaning": Agent(
            role="Data Cleaning Specialist",
            goal="Clean and prepare the dataset for analysis.",
            backstory=CLEANING_PROMPT,
            llm=llm_medium,
            tools=[tool],
            verbose=True,
        ),
        "eda": Agent(
            role="EDA Specialist",
            goal="Explore data patterns, distributions, and correlations.",
            backstory=EDA_PROMPT,
            llm=llm_medium,
            tools=[tool],
            verbose=True,
        ),
        "visualization": Agent(
            role="Visualization Specialist",
            goal="Create insightful charts based on data characteristics.",
            backstory=VIZ_PROMPT,
            llm=llm_medium,
            tools=[tool],
            verbose=True,
        ),
        "statistics": Agent(
            role="Statistical Analysis Expert",
            goal="Run appropriate statistical tests.",
            backstory=STATS_PROMPT,
            llm=llm_medium,
            tools=[tool],
            verbose=True,
        ),
        "feature_engineering": Agent(
            role="Feature Engineering Specialist",
            goal="Create new features, transform existing ones, and prepare the dataset for modelling.",
            backstory=FEATURE_ENG_PROMPT,
            llm=llm_medium,
            tools=[tool],
            verbose=True,
        ),
        "class_imbalance": Agent(
            role="Class Imbalance Specialist",
            goal="Detect and address class imbalance in target variables using resampling, weighting, or synthetic generation.",
            backstory=CLASS_IMBALANCE_PROMPT,
            llm=llm_medium,
            tools=[tool],
            verbose=True,
        ),
        "report": Agent(
            role="Report Generator",
            goal="Synthesize findings into a markdown report.",
            backstory=REPORT_PROMPT,
            llm=llm_long,
            tools=[],
            verbose=True,
        ),
    }


def create_manager(llm):
    return Agent(
        role="Lead Data Analyst",
        goal="Orchestrate data analysis by delegating to the right specialist at the right time.",
        backstory=MANAGER_PROMPT,
        llm=llm,
        allow_delegation=True,
        verbose=True,
    )
