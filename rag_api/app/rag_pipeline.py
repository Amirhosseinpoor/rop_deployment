# rag_api/app/rag_pipeline.py

import os
import re
import requests
import torch
import networkx as nx
from sentence_transformers import CrossEncoder

# LangChain Imports (with deprecation fixes)
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

# --- Global Models and Settings ---
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("MODEL", "deepseek-r1:7b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# This dictionary will hold our initialized models and data
pipeline_resources = {}


# --- Helper Functions ---

def get_reranker():
    """Loads and caches the CrossEncoder model."""
    if "reranker" not in pipeline_resources:
        print("Loading Cross-Encoder model...")
        pipeline_resources["reranker"] = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)
    return pipeline_resources["reranker"]


def build_knowledge_graph(docs):
    """Builds a simple knowledge graph from documents."""
    G = nx.Graph()
    print("Building knowledge graph...")
    for doc in docs:
        entities = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', doc.page_content)
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                G.add_edge(entities[i], entities[i + 1])
    return G


def expand_query_with_hyde(query):
    """Expands the query using the HyDE technique."""
    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": MODEL,
            "prompt": f"Generate a hypothetical answer to: {query}",
            "stream": False
        }).json()
        return f"{query}\n{response.get('response', '')}"
    except Exception as e:
        print(f"Query expansion failed: {str(e)}")
        return query


def retrieve_from_graph(query, G, top_k=3):
    """Retrieves relevant information from the knowledge graph."""
    query_words = query.lower().split()
    matched_nodes = [node for node in G.nodes if any(word in node.lower() for word in query_words)]
    if matched_nodes:
        related_nodes = set()
        for node in matched_nodes:
            related_nodes.update(list(G.neighbors(node)))
        return list(related_nodes)[:top_k]
    return []


# --- Main Pipeline Functions ---

def load_and_process_documents():
    """
    Loads documents from the rag_documents directory, processes them,
    and builds the entire retrieval pipeline. Runs only once at server startup.
    """
    if "retrieval_pipeline" in pipeline_resources:
        print("Retrieval pipeline already loaded.")
        return

    doc_path = "rag_documents"
    print(f"Loading documents from: {doc_path}")

    loaded_docs = []
    if not os.path.exists(doc_path) or not os.path.isdir(doc_path):
        print(f"Warning: Document directory not found at '{doc_path}'. RAG will have no knowledge.")
        pipeline_resources["retrieval_pipeline"] = None
        return

    for filename in os.listdir(doc_path):
        filepath = os.path.join(doc_path, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                loaded_docs.extend(loader.load())
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
                loaded_docs.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                loaded_docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")

    if not loaded_docs:
        print("No documents found to process.")
        pipeline_resources["retrieval_pipeline"] = None
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_chunks = text_splitter.split_documents(loaded_docs)

    print("Creating embeddings and vector store...")
    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    faiss_retriever = FAISS.from_documents(docs_chunks, embeddings).as_retriever(search_kwargs={"k": 5})

    print("Creating BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(docs_chunks)
    bm25_retriever.k = 5

    print("Creating ensemble retriever...")
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    knowledge_graph = build_knowledge_graph(docs_chunks)

    pipeline_resources["retrieval_pipeline"] = {
        "ensemble": ensemble_retriever,
        "knowledge_graph": knowledge_graph,
        "reranker": get_reranker()
    }
    print("✅ RAG pipeline and Knowledge Graph are ready!")


def retrieve_and_rerank_documents(query, chat_history=""):
    """
    Executes the full RAG pipeline: HyDE, Hybrid Search, GraphRAG, Reranking.
    """
    pipeline = pipeline_resources.get("retrieval_pipeline")
    if not pipeline:
        return []

    expanded_query = expand_query_with_hyde(f"{chat_history}\n{query}")
    docs = pipeline["ensemble"].invoke(expanded_query)
    graph_results = retrieve_from_graph(query, pipeline["knowledge_graph"])
    graph_docs = [Document(page_content=node) for node in graph_results]

    combined_docs = list({doc.page_content: doc for doc in graph_docs + docs}.values())  # Remove duplicates

    pairs = [[query, doc.page_content] for doc in combined_docs]
    if not pairs:
        return []

    reranker = pipeline["reranker"]
    scores = reranker.predict(pairs)

    ranked_docs = [doc for _, doc in sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True)]

    return ranked_docs[:3]