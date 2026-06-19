"""ROP-RAG Agentic API microservice.

A hybrid retrieval-augmented-generation service for the ROP clinical assistant:

* **Local RAG** — documents in a folder are chunked, embedded (Ollama) and
  indexed; retrieval combines BM25 (sparse) + FAISS (dense) and is reranked by a
  cross-encoder.
* **Web RAG** — optionally, per query: an LLM crafts a search query, Serper finds
  a page, it is scraped, chunked, embedded into an ephemeral FAISS store and
  reranked. The store is discarded after the query.
* **Generation** — both contexts plus optional image-diagnostic text are fed to
  an OpenAI-compatible chat model with a strict clinical system prompt.

This was already a FastAPI app; here it is reorganised into the standard
``routes`` / ``schemas`` / ``service`` layout and all secrets/paths are moved to
environment variables (no hard-coded keys).
"""
