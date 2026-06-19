"""Hybrid + web RAG pipeline (framework-free).

Faithful port of the original ``rag_api/app/rag_pipeline.py``. Behavioural logic
is unchanged; the differences are operational:

* **No hard-coded secrets.** API keys, base URLs, model ids and the documents
  directory all come from :mod:`config` (env vars). Missing secrets degrade
  gracefully (web search skipped, generation returns a clear error).
* **Lazy, cached resources.** Models/indexes are created on demand and stored in
  a module-level cache, exactly like the original ``pipeline_resources``.
"""
from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import get_settings

# Optional Selenium fallback for JS-rendered pages.
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global resource cache (llm, embeddings, retrieval_pipeline, reranker).
pipeline_resources: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Hybrid retriever (BM25 + dense + cross-encoder rerank).
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """Combines sparse (BM25) and dense (FAISS) retrieval, then reranks."""

    def __init__(self, dense_retriever, documents: list[Document]):
        self.dense_retriever = dense_retriever
        self.documents = documents
        self.bm25 = self._initialize_bm25(documents)
        print("[INIT] Loading CrossEncoder reranker...")
        self.reranker = CrossEncoder(get_settings().reranker_model, device=DEVICE)
        print("[INIT] CrossEncoder ready.")

    def _initialize_bm25(self, documents: list[Document]) -> BM25Okapi:
        texts = [doc.page_content for doc in documents]
        tokenized = [self._tokenize(t) for t in texts]
        print(f"[INIT] BM25 initialized over {len(texts)} chunks.")
        return BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase word tokenisation used by BM25."""
        return re.findall(r"\w+", text.lower())

    def sparse_retrieve(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Top-k BM25 matches with positive scores."""
        scores = self.bm25.get_scores(self._tokenize(query))
        top_indices = np.argsort(scores)[::-1][:k]
        results = [
            (self.documents[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        print(f"[LOCAL][BM25] Retrieved {len(results)} docs for query: {query[:80]}...")
        return results

    def dense_retrieve(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Top-k FAISS matches with similarity scores (falls back gracefully)."""
        try:
            docs_with_scores = self.dense_retriever.vectorstore.similarity_search_with_score(
                query, k=k
            )
            print(f"[LOCAL][Dense] Retrieved {len(docs_with_scores)} docs.")
            return [(doc, float(score)) for doc, score in docs_with_scores]
        except Exception as e:  # noqa: BLE001
            print(f"[LOCAL][Dense] Error in dense retrieval: {e}")
            docs = self.dense_retriever.invoke(query, search_kwargs={"k": k})
            return [(doc, 1.0) for doc in docs]

    def hybrid_retrieve(
        self, query: str, k: int = 10, alpha: float = 0.4
    ) -> list[tuple[Document, float]]:
        """Merge normalised sparse+dense scores: ``alpha*sparse + (1-alpha)*dense``."""
        print(f"[LOCAL][Hybrid] Starting hybrid retrieval for: {query[:80]}...")
        sparse_results = self.sparse_retrieve(query, k=k * 2)
        dense_results = self.dense_retrieve(query, k=k * 2)

        def normalize(results: list[tuple[Document, float]]) -> list[float]:
            if not results:
                return []
            scores = [s for _, s in results]
            mx, mn = max(scores), min(scores)
            if mx == mn:
                return [1.0] * len(scores)
            return [(s - mn) / (mx - mn) for s in scores]

        sparse_norm = normalize(sparse_results)
        dense_norm = normalize(dense_results)

        combined: dict[str, dict[str, Any]] = {}
        for (doc, _), norm in zip(sparse_results, sparse_norm):
            combined[doc.page_content] = {
                "doc": doc,
                "sparse_score": float(norm),
                "dense_score": 0.0,
                "final_score": alpha * float(norm),
            }
        for (doc, _), norm in zip(dense_results, dense_norm):
            if doc.page_content in combined:
                entry = combined[doc.page_content]
                entry["dense_score"] = float(norm)
                entry["final_score"] = alpha * entry["sparse_score"] + (1 - alpha) * float(norm)
            else:
                combined[doc.page_content] = {
                    "doc": doc,
                    "sparse_score": 0.0,
                    "dense_score": float(norm),
                    "final_score": (1 - alpha) * float(norm),
                }

        scored: list[tuple[Document, float]] = []
        for entry in combined.values():
            doc = entry["doc"]
            doc.metadata["sparse_score"] = entry["sparse_score"]
            doc.metadata["dense_score"] = entry["dense_score"]
            doc.metadata["hybrid_score"] = entry["final_score"]
            scored.append((doc, entry["final_score"]))

        scored.sort(key=lambda x: x[1], reverse=True)
        print(f"[LOCAL][Hybrid] Combined docs: {len(scored)}")
        return scored[: k * 2]

    def rerank_documents(
        self, query: str, documents: list[tuple[Document, float]], top_k: int = 5
    ) -> list[Document]:
        """Cross-encoder rerank; on failure keep the input order."""
        if not documents:
            print("[LOCAL][Rerank] No documents to rerank.")
            return []
        docs_only = [doc for doc, _ in documents]
        pairs = [[query, doc.page_content] for doc in docs_only]
        try:
            scores = self.reranker.predict(pairs)
            scored = []
            for doc, score in zip(docs_only, scores):
                doc.metadata["rerank_score"] = float(score)
                scored.append((doc, float(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            final = [doc for doc, _ in scored[:top_k]]
            print(f"[LOCAL][Rerank] Final docs after reranking: {len(final)}")
            return final
        except Exception as e:  # noqa: BLE001
            print(f"[LOCAL][Rerank] Reranking failed, using original order: {e}")
            return [doc for doc, _ in documents[:top_k]]


# --------------------------------------------------------------------------- #
# Local document loading / model factories.
# --------------------------------------------------------------------------- #
def load_documents(directory: str | None = None) -> list[Document]:
    """Load every PDF/TXT/DOCX in ``directory`` (default: configured corpus)."""
    directory = directory or get_settings().documents_dir
    if not os.path.isdir(directory):
        print(f"⚠️ Document directory '{directory}' not found. Skipping document loading.")
        return []

    print(f"[INIT] Loading documents from: {os.path.abspath(directory)}")
    docs: list[Document] = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if filename.endswith(".pdf"):
                docs.extend(PyPDFLoader(filepath).load())
            elif filename.endswith(".txt"):
                docs.extend(TextLoader(filepath).load())
            elif filename.endswith(".docx"):
                docs.extend(Docx2txtLoader(filepath).load())
        except Exception as e:  # noqa: BLE001
            print(f"[INIT] Error loading {filename}: {e}")
            continue
    print(f"[INIT] Loaded {len(docs)} raw documents.")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents (400 chars, 50 overlap) for retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[INIT] Split into {len(chunks)} chunks.")
    return chunks


def get_llm() -> ChatOpenAI:
    """Initialise & cache the OpenAI-compatible chat model."""
    if "llm" not in pipeline_resources:
        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for ChatOpenAI.")
        print(f"[INIT] Initializing ChatOpenAI model: {settings.llm_model} ...")
        pipeline_resources["llm"] = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return pipeline_resources["llm"]


def get_embeddings() -> OllamaEmbeddings:
    """Initialise & cache the Ollama embedding model."""
    if "embeddings" not in pipeline_resources:
        settings = get_settings()
        print(f"[INIT] Loading Ollama Embeddings: {settings.embeddings_model} ...")
        pipeline_resources["embeddings"] = OllamaEmbeddings(
            model=settings.embeddings_model, base_url=settings.ollama_api_url
        )
    return pipeline_resources["embeddings"]


def load_and_process_documents() -> None:
    """Build the full local pipeline (corpus -> chunks -> FAISS -> hybrid)."""
    print("[LIFESPAN] Initializing RAG pipeline...")
    docs = load_documents()
    if not docs:
        print("🛑 No documents found. Skipping pipeline initialization.")
        return

    chunks = split_documents(docs)
    try:
        embeddings = get_embeddings()
    except Exception as e:  # noqa: BLE001
        print(f"🛑 Could not initialise local embeddings (Ollama). Error: {e}")
        return

    print("[INIT] Creating FAISS vector store for dense retrieval...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    except Exception as e:  # noqa: BLE001
        print(f"[INIT] Error creating FAISS index: {e}")
        return

    print("[INIT] Initializing Hybrid Retriever (BM25 + Dense + Reranker)...")
    hybrid_retriever = HybridRetriever(faiss_retriever, chunks)
    pipeline_resources["retrieval_pipeline"] = {"hybrid_retriever": hybrid_retriever}
    pipeline_resources["reranker"] = hybrid_retriever.reranker

    get_llm()  # eager-init the LLM so the first query is fast
    print("✅ Hybrid RAG pipeline (BM25 + Dense + Reranker) is ready!")


def retrieve_and_rerank_documents(query: str, chat_history: str = "") -> list[Document]:
    """Run local hybrid retrieval + rerank; returns [] if pipeline not ready."""
    pipeline = pipeline_resources.get("retrieval_pipeline")
    if not pipeline:
        print("❌ Pipeline not initialized (local hybrid retrieval).")
        return []

    retrieval_query = f"{chat_history}\n{query}" if chat_history else query
    hybrid_retriever: HybridRetriever = pipeline["hybrid_retriever"]
    try:
        hybrid_docs = hybrid_retriever.hybrid_retrieve(retrieval_query, k=15, alpha=0.4)
        final_docs = hybrid_retriever.rerank_documents(retrieval_query, hybrid_docs, top_k=5)
        for i, doc in enumerate(final_docs):
            doc.metadata["hybrid_rank"] = i + 1
            doc.metadata["retrieval_method"] = "hybrid_bm25_dense_reranked"
            doc.metadata.setdefault("origin", "local")
        return final_docs
    except Exception as e:  # noqa: BLE001
        print(f"[LOCAL] Error in hybrid retrieval: {e}")
        return []


# --------------------------------------------------------------------------- #
# Web / agentic RAG.
# --------------------------------------------------------------------------- #
def build_web_search_query(user_query: str, chat_history: str = "") -> str:
    """Ask the LLM for a concise web search query (falls back to the raw query)."""
    llm = get_llm()
    prompt = f"""
You are a search-query generator for a clinical ROP assistant.
Your goal is to create a single, concise Google-style search query that helps
explain or define whatever the user is asking about.

Chat history:
{chat_history}

User question:
{user_query}

Rules:
- Return ONLY the search query string.
- No quotation marks, no extra text, no explanation.
- Focus on ROP/ophthalmology meaning if user asks about a term like "Plus".
"""
    try:
        search_query = llm.invoke(prompt).content.strip()
        print(f"[WEB] LLM-generated search query: {search_query}")
        return search_query
    except Exception as e:  # noqa: BLE001
        print(f"[WEB] Failed to generate search query, using user_query. Error: {e}")
        return user_query


def serper_search(query: str) -> dict:
    """Call the Serper search API; returns {} when no key or on error."""
    settings = get_settings()
    if not settings.serper_api_key:
        print("⚠️ SERPER_API_KEY not set, skipping web search.")
        return {}
    headers = {"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"}
    try:
        print(f"[WEB] Calling Serper with query: {query}")
        resp = requests.post(settings.serper_api_url, headers=headers, json={"q": query}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"[WEB] Serper returned {len(data.get('organic') or [])} organic results.")
        return data
    except Exception as e:  # noqa: BLE001
        print(f"[WEB] Error calling Serper: {e}")
        return {}


def pick_best_result_url(serper_json: dict) -> str | None:
    """Return the first organic result URL, if any."""
    organic = serper_json.get("organic") or []
    if not organic:
        print("[WEB] No organic results to pick from.")
        return None
    url = organic[0].get("link")
    print(f"[WEB] Selected top organic URL: {url}")
    return url


def fetch_page_html(url: str) -> str:
    """Fetch HTML via requests, falling back to headless Selenium if available."""
    try:
        print(f"[WEB] Fetching page via requests: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as e:  # noqa: BLE001
        print(f"[WEB] Requests failed for {url}: {e}")

    if not SELENIUM_AVAILABLE:
        print("[WEB] Selenium not available, giving up on this URL.")
        return ""
    try:
        print(f"[WEB] Falling back to Selenium for URL: {url}")
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:  # noqa: BLE001
        print(f"[WEB] Selenium also failed for {url}: {e}")
        return ""


def extract_clean_text_from_html(html: str) -> str:
    """Strip scripts/nav/etc. and collapse whitespace into a single text blob."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    print(f"[WEB] Extracted text length: {len(text)} characters.")
    return text


def build_web_docs_from_url(url: str) -> list[Document]:
    """Download, clean, and chunk one web page into Documents."""
    raw_text = extract_clean_text_from_html(fetch_page_html(url))
    if not raw_text.strip():
        print(f"[WEB] No text extracted from {url}.")
        return []
    base_doc = Document(page_content=raw_text, metadata={"source": url, "origin": "web"})
    chunks = split_documents([base_doc])
    for d in chunks:
        d.metadata["source"] = url
        d.metadata["origin"] = "web"
    print(f"[WEB] Built {len(chunks)} web document chunks from URL.")
    return chunks


def build_web_vectorstore(docs: list[Document]):
    """Build an ephemeral FAISS store over web docs (lives for one query)."""
    if not docs:
        print("[WEB] No docs provided to build web vectorstore.")
        return None
    print("[WEB] Creating FAISS vector store for web docs...")
    return FAISS.from_documents(docs, get_embeddings())


def rerank_with_cross_encoder(
    query: str, docs_with_scores: list[tuple[Document, float]], top_k: int = 5
) -> list[Document]:
    """Rerank (Document, score) pairs with the shared cross-encoder."""
    if not docs_with_scores:
        print("[WEB][Rerank] No docs to rerank.")
        return []
    reranker: CrossEncoder | None = pipeline_resources.get("reranker")
    if reranker is None:
        print("[WEB][Rerank] Global reranker missing, loading new instance...")
        reranker = CrossEncoder(get_settings().reranker_model, device=DEVICE)
        pipeline_resources["reranker"] = reranker

    docs_only = [doc for doc, _ in docs_with_scores]
    pairs = [[query, doc.page_content] for doc in docs_only]
    try:
        scores = reranker.predict(pairs)
        scored = []
        for doc, s in zip(docs_only, scores):
            doc.metadata["rerank_score"] = float(s)
            doc.metadata.setdefault("origin", "web")
            scored.append((doc, float(s)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]
    except Exception as e:  # noqa: BLE001
        print(f"[WEB][Rerank] Web reranking failed, using original scores: {e}")
        return [doc for doc, _ in docs_with_scores[:top_k]]


def retrieve_from_web(user_query: str, chat_history: str = "", k: int = 10) -> list[Document]:
    """Full ephemeral web-RAG: query -> Serper -> scrape -> FAISS -> rerank."""
    print("[WEB] Starting web RAG pipeline...")
    search_query = build_web_search_query(user_query, chat_history)
    serper_json = serper_search(search_query)
    if not serper_json:
        return []
    url = pick_best_result_url(serper_json)
    if not url:
        return []
    web_docs = build_web_docs_from_url(url)
    if not web_docs:
        return []
    vectorstore = build_web_vectorstore(web_docs)
    if vectorstore is None:
        return []
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(user_query, k=k)
    except Exception as e:  # noqa: BLE001
        print(f"[WEB] Error in web similarity search: {e}")
        return []
    reranked = rerank_with_cross_encoder(user_query, docs_with_scores, top_k=5)
    for i, doc in enumerate(reranked):
        doc.metadata["hybrid_rank"] = i + 1
        doc.metadata["retrieval_method"] = "web_rag"
        doc.metadata["origin"] = "web"
    return reranked


# --------------------------------------------------------------------------- #
# Answer generation.
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """
    You are a warm, clear, and empathetic pediatric retina specialist (ROP).
    You must behave like a real clinician in conversation while strictly controlling when you use medical content.
    Mirror the user's language and level of detail.

    INPUTS YOU WILL RECEIVE
    - [DIAGNOSTIC CONTEXT]: the current findings (Zone, Stage, Plus, Final Decision).
    - [RETRIEVED DOCUMENT CONTEXT]: authoritative ROP references from the clinic’s library and web sources.
      Treat these as the ONLY external facts.
    - [CHAT HISTORY]: prior messages.
    - [USER QUESTION]: the current user message.

    ABSOLUTE RULES
    1) Never invent facts. Only use medical content found in [DIAGNOSTIC CONTEXT] and/or [RETRIEVED DOCUMENT CONTEXT].
       If something is missing, say so briefly and suggest asking the care team.
    2) Keep the tone human and clinician-like: concise, reassuring, plain language first; brief clinical terms second.
    3) Do NOT show citations or your reasoning unless explicitly asked to "Show grounding sources".
    4) Keep formatting simple Markdown.

    INTENT HANDLING (IN THIS PRIORITY ORDER)
    A. Greeting (e.g., “hi”, “hello”, “salam”, “hey”, etc.):
       - Respond with a warm 1–2 sentence greeting and invite a question.
       - Do NOT interpret results or explain the disease unless asked.

    B. Farewell (e.g., “bye”, “goodbye”, “see you”, “khodahafez”):
       - Respond with a polite 1–2 sentence goodbye.
       - Do NOT add clinical explanations.

    C. “Show grounding sources” (or “show sources”):
       - Output ONLY a list titled “Grounding sources:” followed by bullet points of distinct sources
         derived from [RETRIEVED DOCUMENT CONTEXT] metadata (e.g., file/path/title/URL if present).
       - Do NOT add any explanations or commentary.

    D. Section-specific question detected if the user asks about ONE of these keys:
       - {zone, stage, plus, final decision, treatment, recommendation, plan}
       - Respond about that ONE section ONLY. Length target: ~120–150 words.
       - Structure: (1) Plain-language meaning of that section for this case.
         (2) 1–2 brief clinical details. (3) Offer to explain another section if they want.
       - Do NOT discuss other sections unless the user names them.

    E. General disease/condition question (e.g., “what is ROP?”, “what does this mean for my eyes?”, “my results/condition”):
       - Provide a short, integrated explanation grounded in [DIAGNOSTIC CONTEXT] + [RETRIEVED DOCUMENT CONTEXT].
       - Use this format:
         # Summary
         [One short paragraph on what today’s results mean.]
         # Next Steps
         [3–5 concise bullets: what happens next, timelines, typical treatments mentioned in the retrieved context, and follow-up.]
       - If context is insufficient for any item, say so briefly.

    F. Other small talk:
       - Answer naturally in 1–2 sentences; do not introduce medical content unless asked.

    CONTENT BOUNDARIES
    - Do not give emergency directives; if the user describes urgent symptoms, advise contacting their clinical team promptly.
    - No legal or billing advice.
    - No speculation beyond the retrieved sources.

    OUTPUT CHECKLIST
    - If the user greeted or said goodbye, you did NOT include medical explanations.
    - If the user asked a section question, you ONLY discussed that section.
    - If asked to show sources, you ONLY printed a clean bullet list of sources, nothing else.
    - Medical facts appear ONLY if supported by [RETRIEVED DOCUMENT CONTEXT] and/or [DIAGNOSTIC CONTEXT].
    - Tone is warm, concise, and human.
    """


def generate_answer_with_llm(
    query: str,
    context: list[Document],
    chat_history: str = "",
    diagnostic_context_text: str = "",
) -> tuple[str, str]:
    """Generate the grounded answer.

    Returns ``(answer_text, combined_context_str)`` where the second element is
    the exact (possibly truncated) retrieval context handed to the LLM.
    """
    llm = get_llm()

    # Cap history to keep the prompt bounded.
    MAX_HISTORY_LENGTH = 1000
    if len(chat_history) > MAX_HISTORY_LENGTH:
        chat_history = (
            "[... CONVERSATION HISTORY TRUNCATED (Too Long) ...]\n"
            + chat_history[-MAX_HISTORY_LENGTH:]
        )

    if diagnostic_context_text:
        diag_block = f"\nImage-Based Diagnostic Results:\n---\n{diagnostic_context_text}\n---\n"
    else:
        diag_block = "No Image-Based Diagnostic Results provided for this query."

    context_str = "\n---\n".join(doc.page_content for doc in context)
    MAX_CTX_CHARS = 12000
    if len(context_str) > MAX_CTX_CHARS:
        context_str = context_str[:MAX_CTX_CHARS] + "\n[...truncated retrieval context...]"
    print(f"[LLM] Combined context length sent to LLM: {len(context_str)} characters.")

    full_prompt = f"""
    {_SYSTEM_PROMPT}

    [DIAGNOSTIC CONTEXT]:
    {diag_block}

    [RETRIEVED DOCUMENT CONTEXT]:
    {context_str}

    [CHAT HISTORY]:
    {chat_history}

    [USER QUESTION]:
    {query}

    ---
    Now respond appropriately based on the above conversation and intent.
    """
    try:
        print("[LLM] Generating final answer...")
        return llm.invoke(full_prompt).content, context_str
    except Exception as e:  # noqa: BLE001
        return (
            f"Error connecting to LLM: {e}. Please check your OPENAI_API_KEY and network connection.",
            context_str,
        )
