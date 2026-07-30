# doctors_marketplace/agents/retrieval.py
"""Retrieval agents: knowledge base, web search, fetch URL, PubMed."""
from __future__ import annotations

import logging
from typing import Any, Dict

import requests

from .base import BaseAgent, AgentContext, AgentResult
from .registry import register
from ..services import websearch
from ..services.rag import retrieve_context, chunk_text

log = logging.getLogger(__name__)


def _summary(sources, kind):
    if not sources:
        return f"No {kind} results found."
    lines = [f"Found {len(sources)} {kind} result(s):"]
    for i, s in enumerate(sources, 1):
        lines.append(f"- {s.get('title','(untitled)')}: {(s.get('page_content') or '')[:160]}")
    return "\n".join(lines)


@register
class SearchKnowledgeBaseAgent(BaseAgent):
    key = "search_knowledge_base"
    name = "Search knowledge base"
    icon = "book"
    category = "Retrieval"
    description = ("Search THIS doctor's own uploaded knowledge base (their documents, "
                   "guidelines and notes) for passages relevant to the question. Prefer this "
                   "for clinic-specific protocols and anything the doctor has published.")
    input_desc = "query — a focused search phrase."
    output_desc = "Relevant passages from the doctor's documents, added as numbered citations."
    example = ('User: "What is the clinic\'s post-op stone protocol?" → query="post-operative kidney '
               'stone protocol" → returns matching passages from the doctor\'s files, cited [n].')
    stage_label = "Searching the knowledge base"
    produces_sources = True
    run_order = 50
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search phrase."}},
        "required": ["query"],
    }

    def run(self, args, ctx):
        query = (args.get("query") or ctx.query or "").strip()
        try:
            docs = retrieve_context(ctx.doctor, query, k=5)
        except Exception as e:  # noqa: BLE001
            return AgentResult(content=f"Knowledge base unavailable ({e}).", ok=False)
        sources = [{"type": "local",
                    "title": (d.metadata or {}).get("title") or "Document",
                    "url": None, "domain": None,
                    "page_content": d.page_content or ""} for d in docs]
        return AgentResult(content=_summary(sources, "knowledge-base"),
                           sources=sources, display=f"KB: {len(sources)} passage(s)",
                           ok=True)


@register
class SearchWebAgent(BaseAgent):
    key = "search_web"
    name = "Web search"
    icon = "search"
    category = "Retrieval"
    description = ("Search the live web (Google via Serper), read the top medical pages and "
                   "return the most relevant passages. Use for current, external or general "
                   "medical information the knowledge base does not cover.")
    input_desc = "query — a self-contained web search query (resolve pronouns first)."
    output_desc = "Ranked passages from the top web pages, added as numbered citations."
    example = ('User: "latest guidelines for lowering LDL?" → query="2024 LDL cholesterol treatment '
               'guidelines" → returns cited passages from Mayo Clinic, NIH, etc.')
    stage_label = "Searching the web"
    produces_sources = True
    run_order = 55
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Self-contained web query."}},
        "required": ["query"],
    }

    def run(self, args, ctx):
        query = (args.get("query") or ctx.query or "").strip()
        organic = websearch.serper_search(query)[:websearch.WEB_MAX_PAGES]
        docs = websearch.rank_web_pages(query, organic)
        sources = [{"type": "web",
                    "title": (d.metadata or {}).get("title") or websearch.domain_of(d.metadata.get("source", "")),
                    "url": d.metadata.get("source"),
                    "domain": websearch.domain_of(d.metadata.get("source", "")),
                    "page_content": d.page_content or ""} for d in docs]
        return AgentResult(content=_summary(sources, "web"),
                           sources=sources, display=f"Web: {len(sources)} passage(s)",
                           ok=True)


@register
class FetchUrlAgent(BaseAgent):
    key = "fetch_url"
    name = "Read a URL"
    icon = "link"
    category = "Retrieval"
    description = ("Fetch and read a specific web page the user pasted or referenced, so the "
                   "assistant can answer about that exact page. Use when the user gives a link.")
    input_desc = "url — the full http(s) URL to read."
    output_desc = "The readable text of that page, added as a numbered citation."
    example = ('User: "What does this say? https://example.com/rop" → url="https://example.com/rop" '
               '→ returns the page text, cited [n].')
    stage_label = "Reading the page"
    produces_sources = True
    run_order = 55
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Full http(s) URL."}},
        "required": ["url"],
    }

    def run(self, args, ctx):
        url = (args.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return AgentResult(content="Please provide a full http(s) URL.", ok=False)
        text = websearch.scrape_url(url)
        if not text:
            return AgentResult(content=f"Could not read {url} (blocked or empty).", ok=False)
        title = websearch.domain_of(url)
        sources = [{"type": "web", "title": title, "url": url,
                    "domain": websearch.domain_of(url), "page_content": text[:4000]}]
        return AgentResult(content=f"Read {url} ({len(text)} chars).",
                           sources=sources, display=f"Read {title}", ok=True)


@register
class SearchPubMedAgent(BaseAgent):
    key = "search_pubmed"
    name = "PubMed evidence"
    icon = "microscope"
    category = "Retrieval"
    description = ("Search PubMed for peer-reviewed medical literature and return the top "
                   "articles (title, journal, year, abstract). Use when the user wants "
                   "evidence, studies or research backing.")
    input_desc = "query — a PubMed search phrase; optional max_results (default 4)."
    output_desc = "Top PubMed articles with abstracts, added as numbered citations linking to PubMed."
    example = ('User: "Is metformin effective for prediabetes?" → query="metformin prediabetes '
               'prevention" → returns cited PubMed studies.')
    stage_label = "Searching PubMed"
    produces_sources = True
    run_order = 55
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "PubMed search phrase."},
            "max_results": {"type": "integer", "description": "How many articles (1–6)."},
        },
        "required": ["query"],
    }

    _BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def run(self, args, ctx):
        query = (args.get("query") or ctx.query or "").strip()
        n = max(1, min(int(args.get("max_results") or 4), 6))
        try:
            r = requests.get(f"{self._BASE}/esearch.fcgi", params={
                "db": "pubmed", "term": query, "retmax": n, "retmode": "json",
                "sort": "relevance"}, timeout=8)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return AgentResult(content="No PubMed articles found.", ok=True)
            s = requests.get(f"{self._BASE}/esummary.fcgi", params={
                "db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=8)
            s.raise_for_status()
            summ = s.json().get("result", {})
            # Abstracts (one plain-text fetch for all ids).
            try:
                ab = requests.get(f"{self._BASE}/efetch.fcgi", params={
                    "db": "pubmed", "id": ",".join(ids), "rettype": "abstract",
                    "retmode": "text"}, timeout=8).text
            except Exception:
                ab = ""
            abstracts = ab.split("\n\n\n") if ab else []
        except Exception as e:  # noqa: BLE001
            return AgentResult(content=f"PubMed unavailable ({e}).", ok=False)

        sources = []
        for i, pid in enumerate(ids):
            item = summ.get(pid, {})
            authors = ", ".join(a.get("name", "") for a in item.get("authors", [])[:3])
            journal = item.get("fulljournalname") or item.get("source", "")
            year = (item.get("pubdate") or "")[:4]
            title = item.get("title", "PubMed article")
            snippet = abstracts[i][:800] if i < len(abstracts) else ""
            sources.append({
                "type": "web", "title": f"{title} — {journal} ({year})",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/", "domain": "pubmed.ncbi.nlm.nih.gov",
                "page_content": f"{title}. {authors}. {journal} {year}.\n{snippet}"})
        return AgentResult(content=_summary(sources, "PubMed"),
                           sources=sources, display=f"PubMed: {len(sources)} article(s)",
                           ok=True)
