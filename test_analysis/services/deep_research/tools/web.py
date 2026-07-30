"""
T1/T2/T3 — web + knowledge-base retrieval, re-exported from the ROP assistant so
the Deep Research pipeline shares one battle-tested Serper→scrape→FAISS stack.

  web_search(q)              → Serper organic results  [{title, link, snippet}]
  read_and_rank(q, orgs, k)  → scrape + embed-rank → ranked langchain Documents
  kb_retrieve(q, k)          → internal RAG knowledge-base similarity search
  web_enabled()              → is web search turned on?
  domain_of(url)             → bare domain for source cards

All wrapped so a missing/broken web module degrades to "patient-data-only"
grounding instead of crashing the report.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

log = logging.getLogger("test_analysis.deep_research")


def web_enabled() -> bool:
    try:
        from single_rop.chat_service import ENABLE_WEB
        return bool(ENABLE_WEB)
    except Exception:  # noqa: BLE001
        return False


def web_search(query: str) -> List[Dict[str, Any]]:
    if not query:
        return []
    try:
        from single_rop.chat_service import _serper_search
        return _serper_search(query) or []
    except Exception as e:  # noqa: BLE001
        log.info("DR | web_search unavailable (%s)", e)
        return []


def read_and_rank(query: str, organic: List[Dict[str, Any]], k: int = 6) -> List[Any]:
    if not organic:
        return []
    try:
        from single_rop.chat_service import rank_web_pages
        return rank_web_pages(query, organic, k=k) or []
    except Exception as e:  # noqa: BLE001
        log.warning("DR | read_and_rank failed (%s)", e)
        return []


def kb_retrieve(query: str, k: int = 4) -> List[Any]:
    if not query:
        return []
    try:
        from single_rop.chat_service import retrieve_local
        return retrieve_local(query, k=k) or []
    except Exception as e:  # noqa: BLE001
        log.info("DR | kb_retrieve unavailable (%s)", e)
        return []


def domain_of(url: str) -> str:
    try:
        from single_rop.chat_service import _domain
        return _domain(url)
    except Exception:  # noqa: BLE001
        m = re.search(r"https?://([^/]+)", url or "")
        return (m.group(1) if m else "").replace("www.", "")
