
# ============================================
# main.py
# ============================================

import uvicorn
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import (
    load_and_process_documents,
    retrieve_and_rerank_documents,
    generate_answer_with_llm,
    retrieve_from_web,
)


# --- 1. Lifespan for RAG Pipeline Initialization ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the RAG pipeline resources when the application starts
    and cleans up resources when the application shuts down.
    """
    print("[LIFESPAN] Starting up application and loading RAG pipeline...")
    load_and_process_documents()
    yield
    print("[LIFESPAN] Shutting down application...")
    # Optional: Add cleanup logic here if needed


# --- 2. FastAPI App Setup ---

app = FastAPI(
    title="ROP-RAG Agentic API Service",
    version="1.2.0",
    description="Agentic RAG API for ROP assistant (Hybrid Local + Web RAG).",
    lifespan=lifespan,
)

# Allow CORS for development/frontend access
origins = ["*"]  # tighten in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. Pydantic Models ---

class RAGQueryRequest(BaseModel):
    query: str
    chat_history: str = ""
    diagnostic_context_text: str = ""
    use_web: bool = True  # toggle web RAG on/off


class ContextDocument(BaseModel):
    content: str
    source: Optional[str] = None
    sparse_score: Optional[float] = None
    dense_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    hybrid_rank: Optional[int] = None
    retrieval_method: Optional[str] = None
    origin: Optional[str] = None  # "local" or "web"


class RAGQueryResponse(BaseModel):
    """
    Outgoing RAG response.
    You can see exact retrieved corpus from both local and web,
    plus the exact (possibly truncated) combined context sent to the LLM.
    """
    answer: str
    local_context_documents: List[ContextDocument]
    web_context_documents: List[ContextDocument]
    combined_context_truncated: Optional[str] = None


# --- 4. API Endpoints ---

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Agentic RAG API is running."}


@app.post("/query_rag", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: RAGQueryRequest):
    """
    Accepts a user query, retrieves relevant documents from:
      - local hybrid RAG (BM25 + Dense + reranker)
      - web RAG (Serper + scraping + FAISS + reranker) [optional]
    and generates a grounded answer combining both contexts.

    You can see:
      - local_context_documents: exact chunks from local RAG
      - web_context_documents: exact chunks from web pages
      - combined_context_truncated: exact text sent to the LLM (after truncation)
    """
    try:
        print("\n========== NEW /query_rag CALL ==========")
        print(f"[REQUEST] Query: {request.query}")
        print(f"[REQUEST] use_web: {request.use_web}")

        # --- 1. Local hybrid retrieval ---
        local_docs = retrieve_and_rerank_documents(
            query=request.query,
            chat_history=request.chat_history,
        )

        # --- 2. Web RAG (per-query, ephemeral) ---
        web_docs: List = []
        if request.use_web:
            web_docs = retrieve_from_web(
                user_query=request.query,
                chat_history=request.chat_history,
            )
        else:
            print("[WEB] Skipping web RAG (use_web=False).")

        print(f"[COMBINE] Local docs: {len(local_docs)}, Web docs: {len(web_docs)}")

        # --- 3. Build separate context lists (so you see exact corpora) ---

        local_context: List[ContextDocument] = []
        for doc in local_docs:
            md = doc.metadata or {}
            local_context.append(
                ContextDocument(
                    content=doc.page_content,
                    source=md.get("source", "N/A"),
                    sparse_score=md.get("sparse_score"),
                    dense_score=md.get("dense_score"),
                    hybrid_score=md.get("hybrid_score"),
                    rerank_score=md.get("rerank_score"),
                    hybrid_rank=md.get("hybrid_rank"),
                    retrieval_method=md.get("retrieval_method"),
                    origin=md.get("origin", "local"),
                )
            )

        web_context: List[ContextDocument] = []
        for doc in web_docs:
            md = doc.metadata or {}
            web_context.append(
                ContextDocument(
                    content=doc.page_content,
                    source=md.get("source", "N/A"),
                    sparse_score=md.get("sparse_score"),
                    dense_score=md.get("dense_score"),
                    hybrid_score=md.get("hybrid_score"),
                    rerank_score=md.get("rerank_score"),
                    hybrid_rank=md.get("hybrid_rank"),
                    retrieval_method=md.get("retrieval_method"),
                    origin=md.get("origin", "web"),
                )
            )

        all_docs = []
        all_docs.extend(local_docs)
        all_docs.extend(web_docs)

        if not all_docs and not request.diagnostic_context_text:
            llm_context = []
            print("[COMBINE] No retrieval context; answering with diagnostic context only.")
        else:
            llm_context = all_docs

        # --- 4. Generation (returns answer + exact combined context) ---
        answer, used_context_str = generate_answer_with_llm(
            query=request.query,
            context=llm_context,
            chat_history=request.chat_history,
            diagnostic_context_text=request.diagnostic_context_text,
        )

        print("[RESPONSE] Answer generated.")
        return RAGQueryResponse(
            answer=answer,
            local_context_documents=local_context,
            web_context_documents=web_context,
            combined_context_truncated=used_context_str,
        )

    except Exception as e:
        print(f"[ERROR] An error occurred during agentic RAG processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: Failed to process query. Details: {str(e)}",
        )


# /chat endpoint mirrors /query_rag
@app.post("/chat", response_model=RAGQueryResponse)
@app.post("/chat/", response_model=RAGQueryResponse)
async def chat_endpoint(request: RAGQueryRequest):
    """
    Chat-style endpoint that reuses the /query_rag logic.
    """
    request.diagnostic_context_text = (request.diagnostic_context_text or "")[:4000]
    return await query_rag_endpoint(request)


# --- 5. Run Command for Local Development ---
if __name__ == "__main__":
    # Run with:
    #   python -m rag_api.app.main
    # or
    #   uvicorn rag_api.app.main:app --reload
    uvicorn.run("rag_api.app.main:app", host="0.0.0.0", port=8001, reload=True)
