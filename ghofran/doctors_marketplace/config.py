"""Configuration for the Doctors-Marketplace service.

All deployment-specific values (DB, OpenAI credentials, storage roots, the studio
admin key, the embedding model) come from environment variables. No secrets are
hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Immutable configuration view."""

    # --- Persistence ---
    database_url: str = field(
        default_factory=lambda: os.getenv("DM_DATABASE_URL", "sqlite:///./doctors_marketplace.db")
    )
    # Root under which per-doctor FAISS indexes are stored (one subdir per slug).
    vector_root: str = field(
        default_factory=lambda: os.getenv("VECTOR_ROOT", "./dm_doctor_vectors")
    )
    # Root under which uploaded knowledge files are saved.
    knowledge_root: str = field(
        default_factory=lambda: os.getenv("DM_KNOWLEDGE_ROOT", "./dm_doctor_knowledge")
    )

    # --- LLM (OpenAI-compatible) ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    # Optional custom base URL (e.g. an OpenAI-compatible gateway).
    openai_base_url: str = field(default_factory=lambda: os.getenv("BASE_URL", ""))

    # --- Embeddings (HuggingFace) ---
    # Path or model id for the sentence-embedding model used by the RAG layer.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("DM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("DM_EMBEDDING_DEVICE", "cpu"))

    # --- Access control for the studio (doctor CRUD / KB upload) ---
    # Requests to studio endpoints must send this value in the `X-Studio-Key`
    # header. Empty string disables studio endpoints (fail closed).
    studio_api_key: str = field(default_factory=lambda: os.getenv("DM_STUDIO_API_KEY", ""))

    # Network port. Doctors-Marketplace owns 8004.
    port: int = field(default_factory=lambda: int(os.getenv("DM_PORT", "8004")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
