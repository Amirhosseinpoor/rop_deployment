"""Configuration for the ROP-RAG service.

Every credential and tunable is read from the environment. The original code
shipped real API keys as defaults — those are removed here; secret-bearing
fields default to empty and the relevant feature degrades (web search is skipped
if no Serper key; generation errors clearly if no OpenAI key).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Immutable configuration view."""

    # --- LLM (OpenAI-compatible) ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    # --- Embeddings (Ollama) ---
    ollama_api_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    )
    embeddings_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
    )

    # --- Reranker (cross-encoder) ---
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # --- Local document corpus ---
    documents_dir: str = field(
        default_factory=lambda: os.getenv("RAG_DOCUMENTS_DIR", "rag_documents")
    )

    # --- Web RAG (Serper) ---
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    serper_api_url: str = field(
        default_factory=lambda: os.getenv("SERPER_API_URL", "https://google.serper.dev/search")
    )

    # Network port. ROP-RAG owns 8005.
    port: int = field(default_factory=lambda: int(os.getenv("RAG_PORT", "8005")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
