# ============================================
# rag_pipeline.py
# ============================================

import os
import re
import time
import requests
import torch
import numpy as np

from typing import List, Tuple, Optional, Dict, Any

from bs4 import BeautifulSoup

# Optional: Selenium fallback for dynamic pages
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Hybrid RAG imports
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# --- Global Models and Settings ---

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", 'tpsg-TT4pAiTkRvBiG1h16VSeBoARYVfyxrO')
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.metisai.ir/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reranker model (local path)
RERANKER_MODEL = (
    "/home/amir/.cache/huggingface/hub/"
    "models--cross-encoder--ms-marco-MiniLM-L-6-v2/"
    "snapshots/ce0834f22110de6d9222af7a7a03628121708969"
)

# Serper config (your key as default fallback)
SERPER_API_KEY = os.getenv(
    "SERPER_API_KEY",
    "3fd967e3d342dd9e6641649e13850ed8ed2d1ee5"
)
SERPER_API_URL = os.getenv("SERPER_API_URL", "https://google.serper.dev/search")

# Global resource cache
pipeline_resources: Dict[str, Any] = {}


# --- Hybrid RAG Components ---

class HybridRetriever:
    def __init__(self, dense_retriever, documents: List[Document]):
        self.dense_retriever = dense_retriever
        self.documents = documents
        self.bm25 = self._initialize_bm25(documents)
        # Initialize reranker with device
        print("[INIT] Loading CrossEncoder reranker...")
        self.reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
        print("[INIT] CrossEncoder ready.")

    def _initialize_bm25(self, documents: List[Document]):
        """Initialize BM25 with document texts."""
        texts = [doc.page_content for doc in documents]
        tokenized_texts = [self._tokenize(text) for text in texts]
        print(f"[INIT] BM25 initialized over {len(texts)} chunks.")
        return BM25Okapi(tokenized_texts)

    def _tokenize(self, text: str):
        """Simple tokenization for BM25."""
        return re.findall(r"\w+", text.lower())

    def sparse_retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """BM25 sparse retrieval."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))

        print(f"[LOCAL][BM25] Retrieved {len(results)} docs for query: {query[:80]}...")
        return results

    def dense_retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Dense retrieval using existing vector store with scores."""
        try:
            docs_with_scores = self.dense_retriever.vectorstore.similarity_search_with_score(
                query, k=k
            )
            print(f"[LOCAL][Dense] Retrieved {len(docs_with_scores)} docs.")
            return [(doc, float(score)) for doc, score in docs_with_scores]
        except Exception as e:
            print(f"[LOCAL][Dense] Error in dense retrieval: {e}")
            docs = self.dense_retriever.invoke(query, search_kwargs={"k": k})
            return [(doc, 1.0) for doc in docs]

    def hybrid_retrieve(
        self, query: str, k: int = 10, alpha: float = 0.4
    ) -> List[Tuple[Document, float]]:
        """Hybrid retrieval combining sparse and dense results."""
        print(f"[LOCAL][Hybrid] Starting hybrid retrieval for: {query[:80]}...")
        sparse_results = self.sparse_retrieve(query, k=k * 2)
        dense_results = self.dense_retrieve(query, k=k * 2)

        def normalize_scores(results: List[Tuple[Document, float]]):
            if not results:
                return []
            scores = [score for _, score in results]
            max_score = max(scores)
            min_score = min(scores)
            if max_score == min_score:
                return [1.0] * len(scores)
            return [(score - min_score) / (max_score - min_score) for score in scores]

        sparse_scores_norm = normalize_scores(sparse_results)
        dense_scores_norm = normalize_scores(dense_results)

        combined_docs: Dict[str, Dict[str, Any]] = {}

        # Sparse
        for (doc, _), norm_score in zip(sparse_results, sparse_scores_norm):
            combined_docs[doc.page_content] = {
                "doc": doc,
                "sparse_score": float(norm_score),
                "dense_score": 0.0,
                "final_score": alpha * float(norm_score),
            }

        # Dense
        for (doc, _), norm_score in zip(dense_results, dense_scores_norm):
            if doc.page_content in combined_docs:
                combined_docs[doc.page_content]["dense_score"] = float(norm_score)
                combined_docs[doc.page_content]["final_score"] = (
                    alpha * combined_docs[doc.page_content]["sparse_score"]
                    + (1 - alpha) * float(norm_score)
                )
            else:
                combined_docs[doc.page_content] = {
                    "doc": doc,
                    "sparse_score": 0.0,
                    "dense_score": float(norm_score),
                    "final_score": (1 - alpha) * float(norm_score),
                }

        scored_docs: List[Tuple[Document, float]] = []
        for _, scores_dict in combined_docs.items():
            doc = scores_dict["doc"]
            doc.metadata["sparse_score"] = scores_dict["sparse_score"]
            doc.metadata["dense_score"] = scores_dict["dense_score"]
            doc.metadata["hybrid_score"] = scores_dict["final_score"]
            scored_docs.append((doc, scores_dict["final_score"]))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        print(f"[LOCAL][Hybrid] Combined docs: {len(scored_docs)}")
        return scored_docs[: k * 2]

    def rerank_documents(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 5,
    ) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not documents:
            print("[LOCAL][Rerank] No documents to rerank.")
            return []

        docs_only = [doc for doc, _ in documents]
        pairs = [[query, doc.page_content] for doc in docs_only]

        try:
            reranker_scores = self.reranker.predict(pairs)
            scored_docs: List[Tuple[Document, float]] = []

            for doc, score in zip(docs_only, reranker_scores):
                doc.metadata["rerank_score"] = float(score)
                scored_docs.append((doc, float(score)))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in scored_docs[:top_k]]
            print(f"[LOCAL][Rerank] Final docs after reranking: {len(final_docs)}")
            return final_docs

        except Exception as e:
            print(f"[LOCAL][Rerank] Reranking failed, using original order: {e}")
            return [doc for doc, _ in documents[:top_k]]


# --- Helper Functions (Local Docs) ---

def load_documents(directory: str = "rag_documents") -> List[Document]:
    """Loads all documents from the specified directory."""
    docs: List[Document] = []

    if not os.path.isdir(directory):
        print(f"⚠️ Document directory '{directory}' not found. Skipping document loading.")
        return []

    print(f"[INIT] Loading documents from: {os.path.abspath(directory)}")

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
                docs.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                docs.extend(loader.load())
        except Exception as e:
            print(f"[INIT] Error loading {filename}: {e}")
            continue

    print(f"[INIT] Loaded {len(docs)} raw documents.")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"[INIT] Split into {len(chunks)} chunks.")
    return chunks


def get_llm() -> ChatOpenAI:
    """Initializes and caches the ChatOpenAI model."""
    if "llm" not in pipeline_resources:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for ChatOpenAI.")

        print(f"[INIT] Initializing ChatOpenAI model: {LLM_MODEL} with base URL: {OPENAI_BASE_URL}...")
        pipeline_resources["llm"] = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return pipeline_resources["llm"]


def get_embeddings():
    """Loads and caches the Ollama Embeddings model (used for vector search)."""
    if "embeddings" not in pipeline_resources:
        print(f"[INIT] Loading Ollama Embeddings model: {EMBEDDINGS_MODEL} from {OLLAMA_API_URL}...")
        pipeline_resources["embeddings"] = OllamaEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_API_URL,
        )
    return pipeline_resources["embeddings"]


# --- Main Pipeline Functions (Hybrid RAG) ---

def load_and_process_documents():
    """Initializes all components of the hybrid RAG pipeline."""
    print("[LIFESPAN] Initializing RAG pipeline...")
    docs = load_documents()

    if not docs:
        print("🛑 No documents found. Skipping pipeline initialization.")
        return

    docs_chunks = split_documents(docs)

    # 2. Initialize Embeddings and Vector Store
    try:
        embeddings = get_embeddings()
    except Exception as e:
        print(
            f"🛑 Initialization failed: Could not initialize local embeddings (Ollama). "
            f"Check your setup. Error: {e}"
        )
        return

    print("[INIT] Creating FAISS vector store for dense retrieval...")
    try:
        vectorstore = FAISS.from_documents(docs_chunks, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    except Exception as e:
        print(f"[INIT] Error creating FAISS index: {e}")
        return

    # 3. Initialize Hybrid Retriever
    print("[INIT] Initializing Hybrid Retriever (BM25 + Dense + Reranker)...")
    hybrid_retriever = HybridRetriever(faiss_retriever, docs_chunks)

    # 4. Store Resources
    pipeline_resources["retrieval_pipeline"] = {
        "hybrid_retriever": hybrid_retriever,
    }
    pipeline_resources["reranker"] = hybrid_retriever.reranker

    # Ensure LLM is initialized
    get_llm()
    print("✅ Hybrid RAG pipeline (BM25 + Dense + Reranker) is ready!")


def retrieve_and_rerank_documents(query: str, chat_history: str = "") -> List[Document]:
    """Run hybrid retrieval + reranking and return final docs."""
    pipeline = pipeline_resources.get("retrieval_pipeline")
    if not pipeline:
        print("❌ Pipeline not initialized (local hybrid retrieval).")
        return []

    retrieval_query = f"{chat_history}\n{query}" if chat_history else query

    print("[LOCAL] Running Hybrid Retrieval (BM25 + Dense)...")
    hybrid_retriever: HybridRetriever = pipeline["hybrid_retriever"]

    try:
        hybrid_docs_with_scores = hybrid_retriever.hybrid_retrieve(
            retrieval_query, k=15, alpha=0.4
        )
        print(f"[LOCAL] Retrieved {len(hybrid_docs_with_scores)} documents for reranking.")

        print("[LOCAL] Reranking documents with cross-encoder...")
        final_docs = hybrid_retriever.rerank_documents(
            retrieval_query, hybrid_docs_with_scores, top_k=5
        )
        print(f"[LOCAL] Final reranked documents: {len(final_docs)}")

        for i, doc in enumerate(final_docs):
            doc.metadata["hybrid_rank"] = i + 1
            doc.metadata["retrieval_method"] = "hybrid_bm25_dense_reranked"
            doc.metadata.setdefault("origin", "local")

        print("[LOCAL] Top local sources and snippet previews:")
        for i, doc in enumerate(final_docs):
            src = doc.metadata.get("source", "N/A")
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"    [{i+1}] {src} :: {snippet}...")

        return final_docs

    except Exception as e:
        print(f"[LOCAL] Error in hybrid retrieval: {e}")
        return []


# --- Web / Agentic RAG Components ---

def build_web_search_query(user_query: str, chat_history: str = "") -> str:
    """
    Use the LLM to generate a concise web search query based on current
    user query + chat history.
    """
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
    except Exception as e:
        print(f"[WEB] Failed to generate search query from LLM, using user_query. Error: {e}")
        return user_query


def serper_search(query: str) -> dict:
    """
    Call Serper search API and return the raw JSON.
    """
    if not SERPER_API_KEY:
        print("⚠️ SERPER_API_KEY not set, skipping web search.")
        return {}

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    try:
        print(f"[WEB] Calling Serper with query: {query}")
        resp = requests.post(SERPER_API_URL, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        organic_count = len(data.get("organic") or [])
        print(f"[WEB] Serper returned {organic_count} organic results.")
        return data
    except Exception as e:
        print(f"[WEB] Error calling Serper: {e}")
        return {}


def pick_best_result_url(serper_json: dict) -> Optional[str]:
    organic = serper_json.get("organic") or []
    if not organic:
        print("[WEB] No organic results to pick from.")
        return None

    first = organic[0]
    url = first.get("link")
    print(f"[WEB] Selected top organic URL: {url}")
    return url


def fetch_page_html(url: str) -> str:
    """
    Fetch page HTML. Try simple requests first; fallback to Selenium if available.
    """
    try:
        print(f"[WEB] Fetching page via requests: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
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
        time.sleep(3)  # let page render
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        print(f"[WEB] Selenium also failed for {url}: {e}")
        return ""


def extract_clean_text_from_html(html: str) -> str:
    """
    Strip scripts/styles/nav and collapse whitespace. Returns long plain-text string.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    print(f"[WEB] Extracted text length: {len(text)} characters.")
    return text


def build_web_docs_from_url(url: str) -> List[Document]:
    """
    Download, parse, and chunk a single web page into Documents.
    """
    html = fetch_page_html(url)
    raw_text = extract_clean_text_from_html(html)

    if not raw_text.strip():
        print(f"[WEB] No text extracted from {url}.")
        return []

    base_doc = Document(
        page_content=raw_text,
        metadata={
            "source": url,
            "origin": "web",
        },
    )

    # Reuse your splitter
    print("[WEB] Splitting web content into chunks...")
    chunks = split_documents([base_doc])

    for d in chunks:
        d.metadata["source"] = url
        d.metadata["origin"] = "web"

    print(f"[WEB] Built {len(chunks)} web document chunks from URL.")
    return chunks


def build_web_vectorstore(docs: List[Document]):
    """
    Build a simple FAISS vector store over a list of web Documents.
    This is ephemeral and only lives for a single query.
    """
    if not docs:
        print("[WEB] No docs provided to build web vectorstore.")
        return None

    embeddings = get_embeddings()
    print("[WEB] Creating FAISS vector store for web docs...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def rerank_with_cross_encoder(
    query: str, docs_with_scores: List[Tuple[Document, float]], top_k: int = 5
) -> List[Document]:
    """
    Rerank a list of (Document, score) pairs using the global cross-encoder.
    """
    if not docs_with_scores:
        print("[WEB][Rerank] No docs to rerank.")
        return []

    reranker: CrossEncoder = pipeline_resources.get("reranker")
    if reranker is None:
        print("[WEB][Rerank] Global reranker missing, loading new instance...")
        reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
        pipeline_resources["reranker"] = reranker

    docs_only = [doc for doc, _ in docs_with_scores]
    pairs = [[query, doc.page_content] for doc in docs_only]

    try:
        scores = reranker.predict(pairs)
        scored_docs: List[Tuple[Document, float]] = []

        for doc, s in zip(docs_only, scores):
            doc.metadata["rerank_score"] = float(s)
            doc.metadata.setdefault("origin", "web")
            scored_docs.append((doc, float(s)))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in scored_docs[:top_k]]
        print(f"[WEB][Rerank] Final web docs after reranking: {len(final_docs)}")
        return final_docs
    except Exception as e:
        print(f"[WEB][Rerank] Web reranking failed, using original scores: {e}")
        return [doc for doc, _ in docs_with_scores[:top_k]]


def retrieve_from_web(
    user_query: str,
    chat_history: str = "",
    k: int = 10,
) -> List[Document]:
    """
    Full web-RAG pipeline:
    1) LLM search query
    2) Serper
    3) Pick best URL
    4) Scrape + chunk
    5) Build ephemeral FAISS
    6) Similarity search + rerank
    7) Return top Documents (DB is discarded after return)
    """
    print("[WEB] Starting web RAG pipeline...")

    # 1) Build search query
    search_query = build_web_search_query(user_query, chat_history)

    # 2) Call Serper
    serper_json = serper_search(search_query)
    if not serper_json:
        print("[WEB] Serper returned no data.")
        return []

    # 3) Pick best URL
    url = pick_best_result_url(serper_json)
    if not url:
        print("[WEB] No URL selected from Serper results.")
        return []

    # 4) Scrape + chunk
    web_docs = build_web_docs_from_url(url)
    if not web_docs:
        return []

    # 5) Build ephemeral FAISS
    vectorstore = build_web_vectorstore(web_docs)
    if vectorstore is None:
        return []

    # 6) Similarity search
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(user_query, k=k)
        print(f"[WEB] similarity_search_with_score returned {len(docs_with_scores)} docs.")
    except Exception as e:
        print(f"[WEB] Error in web similarity search: {e}")
        return []

    # 7) Rerank
    reranked_docs = rerank_with_cross_encoder(user_query, docs_with_scores, top_k=5)

    for i, doc in enumerate(reranked_docs):
        doc.metadata["hybrid_rank"] = i + 1
        doc.metadata["retrieval_method"] = "web_rag"
        doc.metadata["origin"] = "web"

    print("[WEB] Top web sources and snippet previews:")
    for i, doc in enumerate(reranked_docs):
        src = doc.metadata.get("source", "N/A")
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"    [{i+1}] {src} :: {snippet}...")

    # FAISS vectorstore is local to this function, so it is dropped now
    return reranked_docs


# --- LLM Answer Generation ---

def generate_answer_with_llm(
    query: str,
    context: List[Document],
    chat_history: str = "",
    diagnostic_context_text: str = "",
) -> Tuple[str, str]:
    """
    Generate an answer using the LLM based on retrieved context + diagnostic text.
    Returns:
        (answer_text, context_str_used_for_llm)
    """
    llm = get_llm()

    MAX_HISTORY_LENGTH = 1000
    if len(chat_history) > MAX_HISTORY_LENGTH:
        truncated_history = chat_history[-MAX_HISTORY_LENGTH:]
        chat_history = f"[... CONVERSATION HISTORY TRUNCATED (Too Long) ...]\n{truncated_history}"

    if diagnostic_context_text:
        diag_context_block = f"""
Image-Based Diagnostic Results:
---
{diagnostic_context_text}
---
"""
    else:
        diag_context_block = "No Image-Based Diagnostic Results provided for this query."

    full_context_parts = [doc.page_content for doc in context]
    context_str = "\n---\n".join(full_context_parts)
    MAX_CTX_CHARS = 12000
    if len(context_str) > MAX_CTX_CHARS:
        context_str = context_str[:MAX_CTX_CHARS] + "\n[...truncated retrieval context...]"

    print(f"[LLM] Combined context length sent to LLM: {len(context_str)} characters.")

    system_prompt = """
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

    full_prompt = f"""
    {system_prompt}

    [DIAGNOSTIC CONTEXT]:
    {diag_context_block}

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
        response_content = llm.invoke(full_prompt).content
        return response_content, context_str
    except Exception as e:
        error_msg = (
            "Error connecting to LLM: "
            f"{str(e)}. Please check your OPENAI_API_KEY environment variable and network connection."
        )
        return error_msg, context_str

