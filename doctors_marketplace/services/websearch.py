# doctors_marketplace/services/websearch.py
"""
Generic web-search retrieval for the marketplace assistants.

Mirrors the ROP chatbot's web pipeline (single_rop/chat_service.py) so the two
assistants behave identically:

    Serper search -> scrape top-N pages (BeautifulSoup) -> chunk + embed into an
    ephemeral FAISS store -> similarity-rank the chunks for the query.

Plus ``build_sources_and_context`` which assigns each distinct source a stable
citation id and builds the numbered ``[n]``-prefixed context the LLM sees, so the
model can cite sources inline exactly like the ROP assistant does.
"""
from __future__ import annotations

import logging
import re
import time
from urllib.parse import urlparse
from typing import Any, Dict, List

import requests
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

from .rag import get_embeddings, chunk_text
from .llm import _env

log = logging.getLogger(__name__)   # doctors_marketplace.services.websearch

SERPER_API_KEY = _env("SERPER_API_KEY",
                      default="3fd967e3d342dd9e6641649e13850ed8ed2d1ee5")
SERPER_API_URL = _env("SERPER_API_URL", default="https://google.serper.dev/search")

ENABLE_WEB = _env("DM_ENABLE_WEB", "ROP_ENABLE_WEB", default="1") not in (
    "0", "false", "False", "no")
WEB_MAX_PAGES = int(_env("DM_WEB_MAX_PAGES", "ROP_WEB_MAX_PAGES", default="5"))


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def pretty_title(filename: str) -> str:
    import os
    stem = os.path.splitext(os.path.basename(filename or "document"))[0]
    return re.sub(r"[_\-]+", " ", stem).strip() or "document"


def serper_search(query: str) -> List[Dict[str, Any]]:
    """Return Serper 'organic' results for a query ([] on any failure)."""
    if not SERPER_API_KEY:
        log.info("WEB | SERPER_API_KEY not set — skipping web search.")
        return []
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            SERPER_API_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query}, timeout=8,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic") or []
        log.info("WEB | Serper returned %d organic result(s) in %.2fs",
                 len(organic), time.perf_counter() - t0)
        return organic
    except Exception as e:
        log.warning("WEB | Serper search failed: %s", e)
        return []


def scrape_url(url: str, char_limit: int = 16000) -> str:
    try:
        from bs4 import BeautifulSoup
        t0 = time.perf_counter()
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer",
                         "nav", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        log.info("WEB | scraped %s (%d chars) in %.2fs", url, len(text),
                 time.perf_counter() - t0)
        return text[:char_limit]
    except Exception as e:
        log.warning("WEB | scrape failed for %s: %s", url, e)
        return ""


def rank_web_pages(query: str, organic: List[Dict[str, Any]],
                   k: int = 6) -> List[Document]:
    """Scrape the given result pages, embed and similarity-rank their chunks."""
    docs: List[Document] = []
    for r in organic:
        url = r.get("link")
        if not url:
            continue
        title = r.get("title") or domain_of(url)
        text = scrape_url(url)
        if not text:
            continue
        docs.extend(chunk_text(
            text, metadata={"source": url, "origin": "web", "title": title}))

    if not docs:
        log.info("WEB | no usable web content extracted.")
        return []
    try:
        vs = FAISS.from_documents(docs, embedding=get_embeddings())
        hits = vs.similarity_search(query, k=k)
        log.info("WEB | ranked %d/%d web chunk(s)", len(hits), len(docs))
        return hits
    except Exception as e:
        log.warning("WEB | embedding/ranking failed: %s", e)
        return []


def build_sources_and_context(local_docs: List[Document],
                              web_docs: List[Document]) -> tuple[List[Dict], str]:
    """
    Assign a stable citation id to each distinct source and build the context
    string the LLM sees, each chunk prefixed with its ``[id]`` marker.

    Source dict shape (returned to the UI): {id, type:'local'|'web', title, url,
    domain}. Matches the ROP assistant so the same frontend renderer works.
    """
    sources: List[Dict[str, Any]] = []
    key_to_id: Dict[str, int] = {}
    lines: List[str] = []

    def _source_id(key: str, meta: Dict[str, Any]) -> int:
        if key in key_to_id:
            return key_to_id[key]
        sid = len(sources) + 1
        sources.append({"id": sid, **meta})
        key_to_id[key] = sid
        return sid

    for d in local_docs:
        title = d.metadata.get("title") or pretty_title(d.metadata.get("source", "document"))
        sid = _source_id("local::" + title,
                         {"type": "local", "title": title, "url": None,
                          "domain": None})
        lines.append(f"[{sid}] (document: {title})\n{d.page_content}")

    for d in web_docs:
        url = d.metadata.get("source", "")
        title = d.metadata.get("title") or domain_of(url)
        sid = _source_id("web::" + url,
                         {"type": "web", "title": title, "url": url,
                          "domain": domain_of(url)})
        lines.append(f"[{sid}] (web: {title} — {url})\n{d.page_content}")

    context_str = "\n---\n".join(lines)
    MAX_CTX = 14000
    if len(context_str) > MAX_CTX:
        context_str = context_str[:MAX_CTX] + "\n[...truncated retrieval context...]"
    return sources, context_str
