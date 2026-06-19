"""Thin orchestration layer for the Test-Analysis service.

Re-exports the three capabilities behind small, framework-free functions so the
route layer stays trivial and the heavy modules (pipeline/finders) are imported
lazily — importing this module shouldn't pull in torch/langchain unless a feature
is actually used.
"""
from __future__ import annotations

from .disease_models import predict_hypertension_risk


def predict_hypertension(features: dict) -> str:
    """Run the hypertension model on a dict of patient features.

    ``features`` must contain the 12 model inputs (male, age, ...). Extra keys
    (city/region/insurance) are tolerated. Returns the risk statement string.
    """
    return predict_hypertension_risk(**features)


def chat(message: str, history: list[dict], disease_results=None, personal_information=None):
    """Run one turn of the health chat assistant.

    Imported lazily because the finders module pulls in requests/bs4 and, on
    demand, Selenium.
    """
    from .finders import chat_with_assistant

    return chat_with_assistant(message, history, disease_results, personal_information)


def generate_report(profile_text_summary: str, selected_model: str = "cloud_gpt") -> str:
    """Run the full health-analysis pipeline and return the report text.

    Imported lazily because the pipeline pulls in the (heavy) LangChain RAG stack.
    """
    from .pipeline import run_health_analysis_pipeline

    return run_health_analysis_pipeline(profile_text_summary, selected_model)
