"""Per-doctor RAG (retrieval-augmented generation) over FAISS.

Faithful port of the original ``doctors_marketplace/services/rag.py``. The only
change is decoupling from Django: instead of receiving a Django ``Doctor`` model
and calling ``doctor.vector_dir()``, the functions take an explicit
``vector_dir`` path. The embedding model is read from config rather than being a
hard-coded absolute path.

Each doctor owns a FAISS index built from their uploaded knowledge files. At chat
time we embed the user's query and pull the top-k most similar chunks to ground
the LLM's answer.
"""
from __future__ import annotations

import os
from pathlib import Path

import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

from .config import get_settings


def read_any_text(path: str) -> str:
    """Extract plain text from a PDF, DOCX, or any text file.

    Encoding for plain-text files is auto-detected with ``chardet`` so non-UTF-8
    documents (common with Persian content) still load.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text: list[str] = []
        with open(p, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    if suffix == ".docx":
        try:
            import docx2txt
        except ImportError as e:
            raise RuntimeError("Install docx2txt for .docx support") from e
        return docx2txt.process(path) or ""
    # Fallback: treat as a text file with detected encoding.
    raw = p.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    return raw.decode(enc, errors="ignore")


def chunk_text(text: str) -> list[Document]:
    """Split a document into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.create_documents([text])


def get_embeddings() -> HuggingFaceEmbeddings:
    """Construct the sentence-embedding model from configuration."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_update_index(docs: list[Document], persist_dir: str) -> FAISS:
    """Create a new FAISS index or merge ``docs`` into an existing one."""
    embeddings = get_embeddings()
    if os.path.isdir(persist_dir) and any(Path(persist_dir).glob("*")):
        existing = FAISS.load_local(
            persist_dir, embeddings, allow_dangerous_deserialization=True
        )
        existing.add_documents(docs)
        existing.save_local(persist_dir)
        return existing
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(persist_dir)
    return vs


def index_file(doctor_slug: str, vector_dir: str, abs_file_path: str, title: str) -> tuple[bool, str]:
    """Index one knowledge file into ``vector_dir`` for ``doctor_slug``.

    Returns ``(ok, message)``. ``ok`` is False (with a reason) when the file
    yields no extractable text.
    """
    text = read_any_text(abs_file_path)
    if not text.strip():
        return False, "empty text"
    docs = chunk_text(text)
    for d in docs:
        d.metadata = {"doctor": doctor_slug, "source": abs_file_path, "title": title}
    build_or_update_index(docs, vector_dir)
    return True, "ok"


def retrieve_context(vector_dir: str, query: str, k: int = 4) -> list[Document]:
    """Return the top-``k`` knowledge chunks most similar to ``query``.

    Returns an empty list when the doctor has no index yet, so callers can treat
    "no knowledge base" and "no relevant chunks" uniformly.
    """
    if not os.path.isdir(vector_dir) or not any(Path(vector_dir).glob("*")):
        return []
    embeddings = get_embeddings()
    vs = FAISS.load_local(vector_dir, embeddings, allow_dangerous_deserialization=True)
    return vs.similarity_search(query, k=k)
