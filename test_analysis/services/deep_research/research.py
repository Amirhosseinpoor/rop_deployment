"""
A4 — Multi-channel research + unified graded source pooling (Stage 3).

Evidence is gathered from four channels and fused into ONE numbered source list
that every downstream agent cites, each source tagged with an evidence tier:

  • دادهٔ بیمار (patient) → sources 1..P  (packet order, stable S-ids)
  • دانش‌نامه   (kb)      → curated internal RAG knowledge
  • رهنمود/وب  (web)      → Serper → scrape → embed-rank (current guidance)
  • مطالعه     (lit)      → PubMed E-utilities + Europe PMC (peer-reviewed)

The Question Strategist chose *which* questions to chase and supplied both a web
query and a biomedical `lit_query`; here we run them (web + literature in
parallel), dedupe, rank, and grade. Returns (sources, context, trail) where the
trail feeds the front-end's agentic research display.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import re
from typing import Any, Dict, List, Tuple

from .tools import (web_search, read_and_rank, kb_retrieve, web_enabled,
                    domain_of, lit_search)

log = logging.getLogger("test_analysis.deep_research")

MAX_WEB_QUERIES = 8
WEB_MAX_PAGES = 8
KB_MAX_QUERIES = 10
KB_KEEP = 6
MAX_LIT_QUERIES = 6
LIT_KEEP = 8
CONTEXT_CHAR_CAP = 18000

_WEB_TYPES = {"threshold", "occupational", "workup", "risk", "lifestyle"}
_LIT_TYPES = {"occupational", "workup", "risk", "threshold", "corroboration"}

TIER = {"patient": "دادهٔ بیمار", "kb": "دانش‌نامه", "web": "رهنمود/وب", "lit": "مطالعه"}


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def _select(questions_by_problem, key, types, cap):
    seen: set[str] = set()
    ordered: List[str] = []
    for pass_types in (types, None):
        for qs in questions_by_problem.values():
            for q in qs:
                if pass_types is not None and q.get("type") not in pass_types:
                    continue
                val = q.get(key, "")
                k = _norm(val)
                if val and k not in seen:
                    seen.add(k)
                    ordered.append(val)
                if len(ordered) >= cap:
                    return ordered
    return ordered


def _select_kb_queries(problems, questions_by_problem):
    seen: set[str] = set()
    ordered: List[str] = []
    for p in problems:
        for cand in (p["title"], p.get("mechanism_hint", "")):
            k = _norm(cand)
            if cand and k not in seen:
                seen.add(k)
                ordered.append(cand)
    for qs in questions_by_problem.values():
        for q in qs:
            k = _norm(q["q_fa"])
            if k not in seen:
                seen.add(k)
                ordered.append(q["q_fa"])
            if len(ordered) >= KB_MAX_QUERIES:
                return ordered
    return ordered[:KB_MAX_QUERIES]


def _gather_web(queries: List[str]) -> List[Any]:
    if not queries or not web_enabled():
        return []
    organic: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for results in ex.map(lambda q: web_search(q)[:3], queries):
            for r in results:
                link = r.get("link")
                if link and link not in seen:
                    seen.add(link)
                    organic.append(r)
    organic = organic[:WEB_MAX_PAGES]
    if not organic:
        return []
    docs = read_and_rank(" ; ".join(queries[:5]), organic, k=WEB_MAX_PAGES)
    log.info("DR | web → %d q → %d page(s)", len(queries), len(docs))
    return docs


def _gather_kb(queries: List[str]) -> List[Any]:
    docs, seen = [], set()
    for q in queries:
        for d in kb_retrieve(q, k=3):
            sig = (getattr(d, "page_content", "") or "")[:120]
            if sig and sig not in seen:
                seen.add(sig)
                docs.append(d)
    return docs[:KB_KEEP]


def _gather_lit(queries: List[str]) -> List[Dict[str, Any]]:
    if not queries:
        return []
    refs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for batch in ex.map(lambda q: lit_search(q, retmax=3), queries):
            for r in batch:
                key = r.get("pmid") or r.get("doi") or r.get("title", "")[:60]
                if key and key not in seen:
                    seen.add(key)
                    refs.append(r)
    log.info("DR | literature → %d q → %d ref(s)", len(queries), len(refs[:LIT_KEEP]))
    return refs[:LIT_KEEP]


def gather_sources(packet, problems, questions_by_problem
                   ) -> Tuple[List[Dict[str, Any]], str, Dict[str, int]]:
    sources: List[Dict[str, Any]] = []
    lines: List[str] = []

    # 1) patient facts
    for f in packet["facts"]:
        n = len(sources) + 1
        sources.append({"id": n, "type": "patient", "tier": TIER["patient"],
                        "title": f"{f['label']}: {f['value']}", "url": None,
                        "domain": None, "ref": f.get("ref", ""),
                        "status": f.get("status", "info"), "note": f.get("note", "")})
        note = f" ({f['note']})" if f.get("note") else ""
        lines.append(f"[{n}] (دادهٔ بیمار: {f['label']}) {f['value']}{note}")

    # 2/3/4 — KB, web, literature (web + lit concurrently)
    kb_q = _select_kb_queries(problems, questions_by_problem)
    web_q = _select(questions_by_problem, "search_en", _WEB_TYPES, MAX_WEB_QUERIES)
    lit_q = _select(questions_by_problem, "lit_query", _LIT_TYPES, MAX_LIT_QUERIES)

    kb_docs = _gather_kb(kb_q)
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        fut_web = ex.submit(_gather_web, web_q)
        fut_lit = ex.submit(_gather_lit, lit_q)
        web_docs = fut_web.result()
        lit_refs = fut_lit.result()

    for d in kb_docs:
        n = len(sources) + 1
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or "منبع داخلی"
        sources.append({"id": n, "type": "kb", "tier": TIER["kb"], "title": str(title)[:120],
                        "url": None, "domain": "دانش‌نامهٔ داخلی", "ref": "", "status": "info", "note": ""})
        lines.append(f"[{n}] (دانش‌نامهٔ داخلی: {title})\n{(getattr(d,'page_content','') or '')[:850]}")

    for d in web_docs:
        n = len(sources) + 1
        meta = getattr(d, "metadata", {}) or {}
        url = meta.get("source", "")
        title = meta.get("title") or domain_of(url)
        sources.append({"id": n, "type": "web", "tier": TIER["web"], "title": str(title)[:140],
                        "url": url, "domain": domain_of(url), "ref": "", "status": "info", "note": ""})
        lines.append(f"[{n}] (وب/رهنمود: {title} — {url})\n{(getattr(d,'page_content','') or '')[:850]}")

    for r in lit_refs:
        n = len(sources) + 1
        cite = " · ".join(x for x in (r.get("journal"), r.get("year")) if x)
        sources.append({"id": n, "type": "lit", "tier": TIER["lit"],
                        "title": r.get("title", "")[:180], "url": r.get("url", ""),
                        "domain": r.get("provider", "PubMed"), "ref": "",
                        "pmid": r.get("pmid", ""), "doi": r.get("doi", ""), "cite": cite,
                        "status": "info", "note": ""})
        snip = r.get("snippet") or ""
        lines.append(f"[{n}] (مطالعه — {r.get('provider')}: {r.get('title')} · {cite} · PMID {r.get('pmid') or '—'})\n{snip[:850]}")

    context = "\n---\n".join(lines)
    if len(context) > CONTEXT_CHAR_CAP:
        context = context[:CONTEXT_CHAR_CAP] + "\n[...بریده‌شد...]"

    counts = {
        "patient": sum(1 for s in sources if s["type"] == "patient"),
        "kb": sum(1 for s in sources if s["type"] == "kb"),
        "web": sum(1 for s in sources if s["type"] == "web"),
        "lit": sum(1 for s in sources if s["type"] == "lit"),
    }
    log.info("DR | sources: %(patient)d patient + %(kb)d KB + %(web)d web + %(lit)d lit", counts)
    return sources, context, counts
