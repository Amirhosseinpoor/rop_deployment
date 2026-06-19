"""Pydantic request/response models for the ROP-RAG service."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Incoming query for the RAG endpoints."""

    query: str = Field(..., description="The user's question")
    chat_history: str = Field("", description="Prior conversation text")
    diagnostic_context_text: str = Field(
        "", description="Optional image-based diagnostic findings to ground the answer"
    )
    use_web: bool = Field(True, description="Toggle the per-query web RAG on/off")


class ContextDocument(BaseModel):
    """One retrieved chunk plus its scoring metadata (local or web)."""

    content: str
    source: Optional[str] = None
    sparse_score: Optional[float] = None
    dense_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    hybrid_rank: Optional[int] = None
    retrieval_method: Optional[str] = None
    origin: Optional[str] = Field(None, description="'local' or 'web'")


class RAGQueryResponse(BaseModel):
    """Answer plus the exact corpora used, for full transparency."""

    answer: str
    local_context_documents: list[ContextDocument]
    web_context_documents: list[ContextDocument]
    combined_context_truncated: Optional[str] = Field(
        None, description="Exact (possibly truncated) context string sent to the LLM"
    )
