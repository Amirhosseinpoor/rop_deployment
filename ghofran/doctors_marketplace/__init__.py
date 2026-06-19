"""Doctors-Marketplace microservice.

A marketplace of specialised, RAG-augmented medical chat assistants ("doctors").
Each doctor has a persona, a long system prompt, and an optional private
knowledge base (uploaded files indexed into a per-doctor FAISS vector store).
Patients open a chat session and converse; each reply is grounded in the
doctor's knowledge base when relevant.

Layering:
* :mod:`rag`     — framework-free FAISS retrieval/indexing (ported as-is).
* :mod:`llm`     — thin OpenAI chat client.
* :mod:`service` — business logic over a SQLAlchemy session.
* :mod:`routes`  — the FastAPI surface.
"""
