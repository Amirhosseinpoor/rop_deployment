import os
import re
import requests  # Kept for potential external calls
import torch
import json

# LangChain Imports (with deprecation fixes)
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Global Models and Settings ---
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")  # Reintroduced for local embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", 'tpsg-TT4pAiTkRvBiG1h16VSeBoARYVfyxrO')  # Required for ChatOpenAI (LLM)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL",
                            "https://api.metisai.ir/openai/v1")  # Custom base URL for OpenAI/compatible API
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")  # LLM model name (for OpenAI)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")  # Ollama embeddings model
# CROSS_ENCODER_MODEL and Reranker logic removed for simplicity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# This dictionary will hold our initialized models and data
pipeline_resources = {}


# --- Helper Functions (Simplified) ---

# get_reranker() function removed.
# Graph RAG functions (extract_entities_and_relationships, build_knowledge_graph, retrieve_from_graph) removed.

def load_documents(directory="rag_documents"):
    """Loads all documents from the specified directory."""
    docs = []

    if not os.path.isdir(directory):
        print(f"⚠️ Document directory '{directory}' not found. Skipping document loading.")
        return []

    print(f"Loading documents from: {os.path.abspath(directory)}")

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
            docs.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            docs.extend(loader.load())
    return docs


def split_documents(docs):
    """Splits documents into smaller chunks for retrieval."""
    # Conservative chunk size retained to manage context length
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return text_splitter.split_documents(docs)


def get_llm():
    """Initializes and caches the ChatOpenAI model (used for generation)."""
    if "llm" not in pipeline_resources:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for ChatOpenAI.")

        print(f"Initializing ChatOpenAI model: {LLM_MODEL} with base URL: {OPENAI_BASE_URL}...")
        pipeline_resources["llm"] = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL  # Passing the custom base URL here
        )
    return pipeline_resources["llm"]


def get_embeddings():
    """Loads and caches the Ollama Embeddings model (used for vector search)."""
    if "embeddings" not in pipeline_resources:
        print(f"Loading Ollama Embeddings model: {EMBEDDINGS_MODEL} from {OLLAMA_API_URL}...")
        pipeline_resources["embeddings"] = OllamaEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_API_URL
        )
    return pipeline_resources["embeddings"]


# --- Main Pipeline Functions (Simplified) ---

def load_and_process_documents():
    """Initializes all components of the simple RAG pipeline (Vector Search only)."""
    # 1. Load and Split Documents
    docs = load_documents()

    if not docs:
        print("🛑 No documents found. Skipping pipeline initialization.")
        return

    docs_chunks = split_documents(docs)

    # 2. Initialize Embeddings and Vector Store
    try:
        embeddings = get_embeddings()
    except Exception as e:
        print(f"🛑 Initialization failed: Could not initialize local embeddings (Ollama). Check your setup. Error: {e}")
        return

    print("Creating FAISS vector retriever (Simple RAG)...")
    # Using k=5 for simple vector search retrieval
    faiss_retriever = FAISS.from_documents(docs_chunks, embeddings).as_retriever(search_kwargs={"k": 5})

    # 3. Store Resources
    # Only storing the single FAISS retriever
    pipeline_resources["retrieval_pipeline"] = {
        "retriever": faiss_retriever,
    }
    # Ensure LLM is initialized so get_llm does not fail on first use
    get_llm()
    print("✅ Simple RAG pipeline (Vector Search only) is ready!")


def retrieve_and_rerank_documents(query, chat_history=""):
    """
    Executes the simple RAG pipeline: Vector Search.

    NOTE: Function name maintained for compatibility with application logic,
    but only performs simple retrieval (no rerank/hybrid search/graph RAG).
    """
    pipeline = pipeline_resources.get("retrieval_pipeline")
    if not pipeline:
        return []

    # 1. Prepare Query
    retrieval_query = f"{chat_history}\n{query}"

    # 2. Simple Retrieval (Vector Search only)
    print("Running Simple Vector Retrieval (k=5)...")
    # The retriever is the FAISS retriever stored under the key 'retriever'
    docs = pipeline["retriever"].invoke(retrieval_query)

    # Return the documents directly (top 5 by default, no reranking needed)
    return docs


def generate_answer_with_llm(query, context, chat_history="", diagnostic_context_text=""):
    """
    Generates the final answer using the retrieved context and ChatOpenAI.
    """

    llm = get_llm()
    full_context_parts = []

    # Context Truncation Fix: Limit chat history length to prevent context length errors
    # Max length set to 1000 chars
    MAX_HISTORY_LENGTH = 1000
    if len(chat_history) > MAX_HISTORY_LENGTH:
        # Truncate to keep only the most recent conversation flow
        truncated_history = chat_history[-MAX_HISTORY_LENGTH:]
        chat_history = f"[... CONVERSATION HISTORY TRUNCATED (Too Long) ...]\n{truncated_history}"

    # 1. Add image-based diagnostic results if they are available
    if diagnostic_context_text:
        # Pass the diagnostic context as a separate structured block to guide the LLM's initial output
        diag_context_block = f"""
Image-Based Diagnostic Results:
---
{diagnostic_context_text}
---
"""
    else:
        diag_context_block = "No Image-Based Diagnostic Results provided for this query."

    # 2. Add the retrieved text documents
    full_context_parts = [doc.page_content for doc in context]
    context_str = "\n---\n".join(full_context_parts)
    MAX_CTX_CHARS = 12000
    if len(context_str) > MAX_CTX_CHARS:
        context_str = context_str[:MAX_CTX_CHARS] + "\n[...truncated retrieval context...]"

    system_prompt = (
        "You are a highly skilled, professional, and empathetic medical assistant specializing in Retinopathy of Prematurity (ROP). "
        "Your primary task is to generate a comprehensive, easy-to-understand explanation for the patient/user. "
        "You MUST base your entire response ONLY on the provided [DIAGNOSTIC CONTEXT] and [RETRIEVED DOCUMENT CONTEXT]. "
        "If the retrieved document context is insufficient to fully answer the question, you must state that fact clearly. "
        "Your tone should be professional, empathetic, and reassuring."
    )

    # The full prompt template uses the structure defined in the user's Canvas
    full_prompt = f"""
    {system_prompt}

    [DIAGNOSTIC CONTEXT]:
    {diag_context_block}

    [RETRIEVED DOCUMENT CONTEXT]:
    {context_str}

    [CHAT HISTORY]:
    {chat_history}

    ---
    Based ONLY on the information above, generate your combined response below.

    Follow this strict Markdown structure:

    # Diagnostic Summary

    [Analyze the **DIAGNOSTIC CONTEXT** block. Provide a professional, concise, and empathetic summary of the patient's specific ROP findings: Zone, Stage, Plus status, and the Final Decision/Recommendation for next steps.]

    # User Question

    [State the original user question: '{query}']

    # Comprehensive Explanation

    [This is the main answer. Provide a complete, easy-to-read explanation that directly addresses the user's question. This explanation MUST synthesize the relevant details from **BOTH** the Diagnostic Summary and the **RETRIEVED DOCUMENT CONTEXT** to create a single, unified, grounded answer. The explanation should fully define the condition and contextualize the patient's results within that definition. **Do not include any source citations or thinking process here.**]
    """

    try:
        # Use LangChain's invoke method to generate the final response (replaces requests.post)
        response_content = llm.invoke(full_prompt).content

        return response_content
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}. Please check your **OPENAI_API_KEY** environment variable and network connection."
