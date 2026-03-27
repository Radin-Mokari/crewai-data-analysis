import os
from pathlib import Path
from crewai import LLM
from dotenv import load_dotenv

# Load .env from project root (two levels up from backend/)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


def make_gemini_llm(max_output_tokens: int, thinking_budget: int = 0):
    generation_config = {"max_output_tokens": max_output_tokens}
    if thinking_budget > 0:
        generation_config["thinking"] = {"budget_tokens": thinking_budget}
    return LLM(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        config=generation_config,
    )
