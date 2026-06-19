# RAG API: Global Agentic RAG Microservice

The `rag_api` is a high-performance **FastAPI** service that provides the Mediverse AI platform with distributed, hybrid medical knowledge retrieval and generation.

---

### 1. API Models (`main.py`)
Utilizes Pydantic for strict request/response validation.

- **`RAGQueryRequest`**: Accepts `query`, `chat_history`, `diagnostic_context_text`, and a `use_web` toggle.
- **`RAGQueryResponse`**: Returns the `answer` along with structured `local_context_documents` and `web_context_documents` (including metadata like hybrid scores and rerank scores) for full transparency.

---

### 2. Core Logic & AI Pipeline (`rag_pipeline.py`)
This module implements a state-of-the-art **Hybrid RAG** architecture.

#### **Step-by-Step Data Flow:**
1.  **Local Retrieval (`HybridRetriever`)**:
    - **Sparse**: BM25 algorithm (via `Rank-BM25`) indexes text chunks for keyword matching.
    - **Dense**: FAISS vector store with **Ollama** embeddings (`nomic-embed-text`) for semantic matching.
    - **Hybrid Score**: Weighted combination of Sparse and Dense scores.
2.  **Reranking**:
    - **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    - **Logic**: Re-evaluates the top 15 candidates for direct relevance to the query.
3.  **Agentic Web RAG (`retrieve_from_web`)**:
    - **Query Generation**: LLM transforms the user's question into an optimized Google search query.
    - **Search**: Executes via **Serper API**.
    - **Scraping**: Fetches top result HTML (using **Selenium** if dynamic rendering is required).
    - **Ephemeral Index**: Chunks the web text and builds a temporary FAISS index for immediate retrieval.
4.  **Generation**:
    - Combines local context, web context, and diagnostic context into a massive system prompt for the final LLM synthesis.

#### **I/O Example:**
- **Input**: `{"query": "What is Stage 3 ROP?", "use_web": true}`
- **Output**: Grounded answer + list of retrieved PDF sources + list of web URLs used.

---

### 3. Routers & Handlers (`main.py`)
- **`lifespan`**: Initializes the RAG pipeline resources (Loading PDFs, initializing BM25) on application startup.
- **`query_rag_endpoint`**: The primary async handler that coordinates local retrieval, optional web scraping, and LLM generation.
- **`health_check`**: Returns system status and RAG readiness.

---

### 4. Routing
| Method | Endpoint | Logic |
| :--- | :--- | :--- |
| `POST` | `/query_rag` | Full RAG pipeline execution. |
| `POST` | `/chat` | Conversational RAG (mirror of `/query_rag`). |
| `GET` | `/health` | Service status monitor. |

---

*Note: This is a headless backend service. No frontend templates are present.*
