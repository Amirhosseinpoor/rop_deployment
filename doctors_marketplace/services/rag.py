# doctors_marketplace/services/rag.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
import chardet

# ---- Loaders (unchanged) ----
def read_any_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        text = []
        with open(p, "rb") as f:
            r = PdfReader(f)
            for page in r.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    elif p.suffix.lower() in [".docx"]:
        try:
            import docx2txt
        except ImportError:
            raise RuntimeError("Install python-docx/docx2txt for .docx support")
        return docx2txt.process(path) or ""
    else:
        raw = p.read_bytes()
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        return raw.decode(enc, errors="ignore")

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.create_documents([text])

# ---- Embeddings (HuggingFace) ----
MODEL_PATH = "/home/amir/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ---- Indexing ----
def build_or_update_index(docs: List[Document], persist_dir: str):
    embeddings = get_embeddings()
    if os.path.isdir(persist_dir) and any(Path(persist_dir).glob("*")):
        existing = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        existing.add_documents(docs)
        existing.save_local(persist_dir)
        return existing
    else:
        vs = FAISS.from_documents(docs, embedding=embeddings)
        vs.save_local(persist_dir)
        return vs

def index_file_for_doctor(doctor, abs_file_path: str, title: str):
    text = read_any_text(abs_file_path)
    if not text.strip():
        return False, "empty text"
    docs = chunk_text(text)
    for d in docs:
        d.metadata = {"doctor": doctor.slug, "source": abs_file_path, "title": title}
    vs_dir = doctor.vector_dir()
    build_or_update_index(docs, vs_dir)
    return True, "ok"

def retrieve_context(doctor, query: str, k: int = 4) -> List[Document]:
    vs_dir = doctor.vector_dir()
    if not os.path.isdir(vs_dir) or not any(Path(vs_dir).glob("*")):
        return []
    embeddings = get_embeddings()
    vs = FAISS.load_local(vs_dir, embeddings, allow_dangerous_deserialization=True)
    return vs.similarity_search(query, k=k)
