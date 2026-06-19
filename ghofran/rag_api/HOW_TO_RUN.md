# How to Run ROP-RAG Service

A hybrid retrieval-augmented-generation API for the ROP clinical assistant. It
combines a **local** corpus (BM25 + FAISS dense + cross-encoder rerank) with an
optional **web** RAG (Serper search → scrape → ephemeral FAISS → rerank), then
generates a grounded answer with an OpenAI-compatible chat model.

## Prerequisites
- Python 3.10+
- **Ollama** running locally with an embedding model pulled (default
  `nomic-embed-text`):
  ```bash
  ollama pull nomic-embed-text
  ```
- An **OpenAI-compatible API key** for answer generation.
- *(Optional)* A **Serper** API key for web RAG.
- First run downloads the cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`).
- A documents directory (`rag_documents/`, bundled) for the local corpus.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration (environment variables)

| Variable             | Default                                  | Description                                  |
|----------------------|------------------------------------------|----------------------------------------------|
| `OPENAI_API_KEY`     | *(required for answers)*                  | OpenAI-compatible API key.                   |
| `OPENAI_BASE_URL`    | `https://api.openai.com/v1`              | LLM endpoint (set to your gateway if any).   |
| `LLM_MODEL`          | `gpt-4o-mini`                            | Chat model name.                             |
| `OLLAMA_API_URL`     | `http://localhost:11434`                | Ollama server URL (embeddings).              |
| `EMBEDDINGS_MODEL`   | `nomic-embed-text:latest`               | Ollama embedding model.                      |
| `CROSS_ENCODER_MODEL`| `cross-encoder/ms-marco-MiniLM-L-6-v2`  | Reranker model id or local path.             |
| `RAG_DOCUMENTS_DIR`  | `rag_documents`                         | Folder of PDF/TXT/DOCX for the local corpus. |
| `SERPER_API_KEY`     | *(empty = web RAG disabled)*            | Serper search key.                           |
| `SERPER_API_URL`     | `https://google.serper.dev/search`      | Serper endpoint.                             |
| `RAG_PORT`           | `8005`                                   | Port for the convenience launcher.           |

```bash
export OPENAI_API_KEY=sk-...
export RAG_DOCUMENTS_DIR="$(pwd)/rag_documents"   # the bundled corpus
```

## Run
From inside `ghofran/`:
```bash
cd ..
uvicorn rag_api.routes:app --host 0.0.0.0 --port 8005
```
Or self-launch (honours `RAG_PORT`):
```bash
cd rag_api
python routes.py
```
The local pipeline is built at start-up (you'll see `✅ Hybrid RAG pipeline ...`
once ready). Interactive docs: <http://localhost:8005/docs>

## Test

**1. Liveness**
```bash
curl http://localhost:8005/health
```

**2. Query (local only — fast, no Serper needed)**
```bash
curl -X POST http://localhost:8005/query_rag -H 'Content-Type: application/json' -d '{
  "query": "What does Plus disease mean in ROP?",
  "use_web": false
}'
```

**3. Query with web RAG + diagnostic context**
```bash
curl -X POST http://localhost:8005/chat -H 'Content-Type: application/json' -d '{
  "query": "What should I do next?",
  "use_web": true,
  "diagnostic_context_text": "Zone 2, Stage 3, Plus present, Final decision: Treatment"
}'
```

The response includes `answer`, `local_context_documents`, `web_context_documents`,
and `combined_context_truncated` (the exact context sent to the LLM).

## Notes
- If Ollama isn't running, local retrieval is skipped and answers rely on web RAG
  and/or the diagnostic context only.
- If `SERPER_API_KEY` is unset, web RAG is silently skipped (`use_web` has no
  effect) and only the local corpus is used.
```
