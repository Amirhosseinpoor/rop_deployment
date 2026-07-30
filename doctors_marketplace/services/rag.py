# doctors_marketplace/services/rag.py
"""
Per-doctor Retrieval-Augmented Generation.

Pipeline overview
-----------------
1. Loaders        -> read_any_text() turns PDF / DOCX / TXT into plain text.
2. Chunking       -> token-aware RecursiveCharacterTextSplitter (tiktoken),
                     tuned for mixed Persian/English medical notes.
3. Embeddings     -> GAPGPT `text-embedding-3-large` (3072-dim) via the
                     OpenAI-compatible API. No local model / torch required,
                     which keeps `runserver` lightweight and robust.
4. Index          -> FAISS, one store per doctor under media/doctor_vectors/.
5. Retrieval      -> MMR search (relevance + diversity) with a score floor.
"""
import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_core.embeddings import Embeddings

from openai import OpenAI
import chardet

from .llm import API_KEY, BASE_URL, _env

log = logging.getLogger(__name__)

EMBED_MODEL = _env("GAPGPT_EMBED_MODEL", "OPENAI_EMBED_MODEL",
                   default="text-embedding-3-large")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """
    NFKC-normalise text. Crucial for Persian/Arabic PDFs, whose glyphs are often
    stored as Unicode *presentation forms* (e.g. U+FE97) — NFKC maps those back
    to the base letters (U+062A) that a user actually types, so document and
    query embeddings live in the same character space.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‌", " ")          # ZWNJ -> space
    text = re.sub(r"[ \t]{2,}", " ", text)        # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank lines
    return text.strip()


def _read_pdf(path: Path) -> str:
    """Prefer PyMuPDF (much better Persian handling); fall back to PyPDF2."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(path)) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception:
        from PyPDF2 import PdfReader
        with open(path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_any_text(path: str) -> str:
    """Extract plain, normalised text from a PDF, DOCX or text-like file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return normalize_text(_read_pdf(p))
    if suffix in (".docx", ".doc"):
        try:
            import docx2txt
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install docx2txt for .docx support") from e
        return normalize_text(docx2txt.process(str(p)) or "")
    # Fallback: decode as text with charset detection.
    raw = p.read_bytes()
    enc = (chardet.detect(raw).get("encoding") or "utf-8")
    return normalize_text(raw.decode(enc, errors="ignore"))


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Persian/English sentence + clause separators, coarse -> fine.
_SEPARATORS = ["\n\n", "\n", "؟", "!", ".", "؛", "،", " ", ""]


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Token-aware splitter (~320 tokens, 60-token overlap). Token counting keeps
    chunks consistent across Persian and English, where character length is a
    poor proxy for model context size.
    """
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=320,
            chunk_overlap=60,
            separators=_SEPARATORS,
        )
    except Exception:  # pragma: no cover - tiktoken unavailable
        return RecursiveCharacterTextSplitter(
            chunk_size=1100, chunk_overlap=200, separators=_SEPARATORS,
        )


def chunk_text(text: str, metadata: dict | None = None) -> List[Document]:
    """Normalise whitespace, split into chunks, drop tiny noise fragments."""
    text = "\n".join(line.rstrip() for line in text.splitlines())
    docs = _build_splitter().create_documents([text], metadatas=[metadata or {}])
    return [d for d in docs if len(d.page_content.strip()) >= 30]


# --------------------------------------------------------------------------- #
# Embeddings (GAPGPT, OpenAI-compatible)
# --------------------------------------------------------------------------- #
class GapGPTEmbeddings(Embeddings):
    """LangChain Embeddings backed by the GAPGPT embeddings endpoint."""

    def __init__(self, model: str = EMBED_MODEL, batch_size: int = 64):
        if not API_KEY:
            raise RuntimeError("No embeddings API key (GAPGPT_API_KEY / OPENAI_API_KEY)")
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[i:i + self.batch_size]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


_embeddings_singleton: GapGPTEmbeddings | None = None


def get_embeddings() -> GapGPTEmbeddings:
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = GapGPTEmbeddings()
    return _embeddings_singleton


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def _has_index(persist_dir: str) -> bool:
    return os.path.isfile(os.path.join(persist_dir, "index.faiss"))


def build_or_update_index(docs: List[Document], persist_dir: str) -> FAISS:
    embeddings = get_embeddings()
    if _has_index(persist_dir):
        try:
            existing = FAISS.load_local(
                persist_dir, embeddings, allow_dangerous_deserialization=True
            )
            existing.add_documents(docs)
            existing.save_local(persist_dir)
            return existing
        except Exception as e:
            # Most likely an embedding-dimension change from an older model.
            # Rebuild this store from scratch with the current embeddings.
            log.warning("Rebuilding FAISS index at %s (incompatible existing index: %s)",
                        persist_dir, e)
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(persist_dir)
    return vs


def index_file_for_doctor(doctor, abs_file_path: str, title: str):
    """Read, chunk and add a single knowledge file to the doctor's index."""
    text = read_any_text(abs_file_path)
    if not text.strip():
        return False, "empty text"
    docs = chunk_text(text, metadata={
        "doctor": doctor.slug, "source": os.path.basename(abs_file_path), "title": title,
    })
    if not docs:
        return False, "no usable chunks"
    build_or_update_index(docs, doctor.vector_dir())
    return True, f"indexed {len(docs)} chunks"


def rebuild_doctor_index(doctor) -> tuple[bool, str]:
    """Rebuild a doctor's whole index from all of their knowledge files."""
    vs_dir = doctor.vector_dir()
    # Clear the old store so a stale-dimension index can't linger.
    for name in ("index.faiss", "index.pkl"):
        f = os.path.join(vs_dir, name)
        if os.path.isfile(f):
            os.remove(f)

    all_docs: List[Document] = []
    for item in doctor.knowledge_items.all():
        try:
            path = item.file.path
        except Exception:
            continue
        if not path or not os.path.isfile(path):
            continue
        text = read_any_text(path)
        if not text.strip():
            continue
        all_docs.extend(chunk_text(text, metadata={
            "doctor": doctor.slug,
            "source": os.path.basename(path),
            "title": item.title,
        }))

    if not all_docs:
        return False, "no content to index"
    vs = FAISS.from_documents(all_docs, embedding=get_embeddings())
    vs.save_local(vs_dir)
    return True, f"rebuilt with {len(all_docs)} chunks"


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def retrieve_context(doctor, query: str, k: int = 5) -> List[Document]:
    """
    MMR retrieval (relevance + diversity). The query is NFKC-normalised so it
    matches normalised document text. Returns [] on any failure
    (missing/incompatible index, API error) so chat degrades gracefully.

    No hard score-floor is applied: FAISS relevance scores are unreliable across
    embedding models, and the assistant prompt already instructs the model to
    ignore retrieved snippets that aren't relevant.
    """
    vs_dir = doctor.vector_dir()
    if not _has_index(vs_dir):
        return []
    try:
        embeddings = get_embeddings()
        vs = FAISS.load_local(vs_dir, embeddings, allow_dangerous_deserialization=True)
        mmr = vs.max_marginal_relevance_search(
            normalize_text(query), k=k, fetch_k=max(k * 5, 20))
        ordered, seen = [], set()
        for d in mmr:
            key = d.page_content.strip()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(d)
        return ordered[:k]
    except Exception as e:
        log.warning("retrieve_context failed for %s: %s", doctor.slug, e)
        return []
