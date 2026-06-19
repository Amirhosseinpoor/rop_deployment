"""Configuration for the Test-Analysis (health) service.

All credentials, model locations and data paths come from environment variables.
The service has three fairly independent features, each with its own external
dependencies; the relevant config is grouped and documented below. No secrets
are hard-coded (the original shipped real keys — removed here).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


# The hypertension model bundled in the ``model/`` directory next to this
# package, resolved absolutely so it is found regardless of the working directory.
_BUNDLED_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


@dataclass(frozen=True)
class Settings:
    """Immutable configuration view."""

    # --- Hypertension model (feature 1) ---
    hypertension_model_path: str = field(
        default_factory=lambda: os.getenv(
            "HYPERTENSION_MODEL_PATH",
            os.path.join(os.getenv("MODEL_DIR", _BUNDLED_MODEL_DIR), "best_rf_hypertension_model.joblib"),
        )
    )

    # --- Chat assistant LLM (feature 2): GapGPT / OpenAI-compatible ---
    chat_api_key: str = field(default_factory=lambda: os.getenv("GAPGPT_API_KEY", ""))
    chat_base_url: str = field(default_factory=lambda: os.getenv("GAPGPT_BASE_URL", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("GAPGPT_MODEL", "gpt-5-nano"))

    # --- Report pipeline LLM (feature 3): Metis / OpenAI-compatible ---
    pipeline_api_key: str = field(default_factory=lambda: os.getenv("METIS_API_KEY", ""))
    pipeline_base_url: str = field(default_factory=lambda: os.getenv("BASE_URL", ""))
    pipeline_model: str = field(default_factory=lambda: os.getenv("MODEL_NAME_LLM", "gpt-4o-mini"))
    # Optional local model (Ollama) alternative for the pipeline LLM.
    ollama_api_url: str = field(default_factory=lambda: os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
    local_model_name: str = field(default_factory=lambda: os.getenv("LOCAL_MODEL_NAME", "llama3"))

    # --- RAG resources for the report pipeline ---
    embedding_model: str = field(
        default_factory=lambda: os.getenv("HEALTH_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    knowledge_base_dir: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
    )
    drugstores_csv: str = field(
        default_factory=lambda: os.getenv("DRUGSTORES_CSV", "data_csv/drugstores.csv")
    )

    # Network port. Test-Analysis owns 8006.
    port: int = field(default_factory=lambda: int(os.getenv("HEALTH_PORT", "8006")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
