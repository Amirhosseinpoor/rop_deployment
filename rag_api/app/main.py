import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import functions from your RAG pipeline
# FIX: Changed import from absolute path (rag_api.app.rag_pipeline) to
# a relative path (.app.rag_pipeline) to resolve ModuleNotFoundError when running main.py directly.
from .rag_pipeline import (
    load_and_process_documents,
    retrieve_and_rerank_documents,
    generate_answer_with_llm
)


# --- 1. Lifespan for RAG Pipeline Initialization ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the RAG pipeline resources when the application starts
    and cleans up resources when the application shuts down.
    """
    print("Starting up application and loading RAG pipeline...")
    # This calls load_and_process_documents() from rag_pipeline.py
    load_and_process_documents()
    yield
    print("Shutting down application...")
    # Optional: Add cleanup logic here if needed (e.g., closing connections)


# --- 2. FastAPI App Setup ---


app = FastAPI(
    title="ROP-RAG API Service",
    version="1.0.0",
    description="...",
    # redirect_slashes=True is the FastAPI default; we’ll add both routes below anyway.
)

# Allow CORS for development/frontend access
origins = [
    "*",  # Be restrictive in production!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. Pydantic Models for API Data Validation ---

class RAGQueryRequest(BaseModel):
    query: str
    chat_history: str = ""
    diagnostic_context_text: str = ""


class RAGQueryResponse(BaseModel):
    """
    Defines the structure for the outgoing RAG response.
    """
    answer: str
    context_documents: list[dict]


# --- 4. API Endpoints ---

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "RAG API is running."}


@app.post("/query_rag", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: RAGQueryRequest):
    """
    Accepts a user query, retrieves relevant documents, and generates a grounded answer.
    """
    try:
        # Step 1: Retrieval and Reranking
        context_docs = retrieve_and_rerank_documents(
            query=request.query,
            chat_history=request.chat_history
        )

        # Prepare context documents for the response model
        response_context = [
            {"content": doc.page_content, "source": doc.metadata.get("source", "N/A"),
             "score": doc.metadata.get("score", 0.0)}
            for doc in context_docs
        ]

        if not context_docs and not request.diagnostic_context_text:
            # Fallback if no context is found (allowing LLM to answer only if image data exists)
            llm_context = []
        else:
            llm_context = context_docs

        # Step 2: Generation
        # IMPORTANT: Pass the new diagnostic_context_text parameter to the LLM function
        answer = generate_answer_with_llm(
            query=request.query,
            context=llm_context,
            chat_history=request.chat_history,
            diagnostic_context_text=request.diagnostic_context_text  # <-- The essential change
        )

        return RAGQueryResponse(
            answer=answer,
            context_documents=response_context
        )

    except Exception as e:
        print(f"An error occurred during RAG processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: Failed to process query. Details: {str(e)}"
        )


# New endpoint for /chat to resolve the 404 error
@app.post("/chat")
@app.post("/chat/")
async def chat_endpoint(request: RAGQueryRequest):
    # Hard cap the size to avoid accidental blow-ups
    request.diagnostic_context_text = (request.diagnostic_context_text or "")[:4000]
    return await query_rag_endpoint(request)

# --- 5. Run Command for Local Development ---
if __name__ == "__main__":
    # Note: When running with uvicorn directly, you might need to adjust the import path
    # and call uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8001)
