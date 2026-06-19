"""FastAPI layer for the ROP-RAG service.

Ported from the original ``app/main.py``. Endpoints:

* ``GET  /health``     — liveness.
* ``POST /query_rag``  — hybrid local + optional web RAG, returns answer + corpora.
* ``POST /chat``       — alias of ``/query_rag`` that caps the diagnostic context.

Exports ``router`` (mountable) and ``app`` (runnable via ``uvicorn routes:app``).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import service
from .config import get_settings
from .schemas import ContextDocument, RAGQueryRequest, RAGQueryResponse

router = APIRouter(tags=["rag"])


def _to_context_documents(docs: list, default_origin: str) -> list[ContextDocument]:
    """Map LangChain Documents to API ``ContextDocument`` models."""
    out: list[ContextDocument] = []
    for doc in docs:
        md = doc.metadata or {}
        out.append(
            ContextDocument(
                content=doc.page_content,
                source=md.get("source", "N/A"),
                sparse_score=md.get("sparse_score"),
                dense_score=md.get("dense_score"),
                hybrid_score=md.get("hybrid_score"),
                rerank_score=md.get("rerank_score"),
                hybrid_rank=md.get("hybrid_rank"),
                retrieval_method=md.get("retrieval_method"),
                origin=md.get("origin", default_origin),
            )
        )
    return out


@router.get("/health", summary="Liveness probe")
def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Agentic RAG API is running."}


@router.post("/query_rag", response_model=RAGQueryResponse, summary="Hybrid local + web RAG query")
async def query_rag_endpoint(request: RAGQueryRequest) -> RAGQueryResponse:
    """Answer a query by combining local hybrid retrieval and (optional) web RAG.

    The response exposes the exact local and web corpora plus the combined
    context string sent to the LLM, for full auditability.
    """
    try:
        local_docs = service.retrieve_and_rerank_documents(
            query=request.query, chat_history=request.chat_history
        )
        web_docs: list = []
        if request.use_web:
            web_docs = service.retrieve_from_web(
                user_query=request.query, chat_history=request.chat_history
            )

        all_docs = [*local_docs, *web_docs]
        # If nothing was retrieved and no diagnostic context exists, answer with
        # an empty context (the LLM then explains it lacks grounding).
        llm_context = all_docs if (all_docs or request.diagnostic_context_text) else []

        answer, used_context_str = service.generate_answer_with_llm(
            query=request.query,
            context=llm_context,
            chat_history=request.chat_history,
            diagnostic_context_text=request.diagnostic_context_text,
        )
        return RAGQueryResponse(
            answer=answer,
            local_context_documents=_to_context_documents(local_docs, "local"),
            web_context_documents=_to_context_documents(web_docs, "web"),
            combined_context_truncated=used_context_str,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: Failed to process query. Details: {e}",
        )


@router.post("/chat", response_model=RAGQueryResponse, summary="Chat alias of /query_rag")
@router.post("/chat/", response_model=RAGQueryResponse, include_in_schema=False)
async def chat_endpoint(request: RAGQueryRequest) -> RAGQueryResponse:
    """Chat-style endpoint that caps the diagnostic context then reuses /query_rag."""
    request.diagnostic_context_text = (request.diagnostic_context_text or "")[:4000]
    return await query_rag_endpoint(request)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Build the local RAG pipeline at start-up (tolerant of missing resources)."""
    print("[LIFESPAN] Starting up application and loading RAG pipeline...")
    service.load_and_process_documents()
    yield
    print("[LIFESPAN] Shutting down application...")


app = FastAPI(
    title="ROP-RAG Agentic API Service",
    version="1.2.0",
    description="Agentic RAG API for ROP assistant (Hybrid Local + Web RAG).",
    lifespan=_lifespan,
)
# CORS open for development; tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_settings().port)
