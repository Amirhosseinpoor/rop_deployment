"""
lit_search — biomedical literature retrieval from **NCBI PubMed E-utilities** and
**Europe PMC**. Free, key-less, and a genuinely different class of source than a
Google scrape: peer-reviewed guidelines, systematic reviews and primary studies
with PMIDs/DOIs the report can cite precisely.

Best-effort by design: short timeouts, both providers tried, and any failure
(network, geo-block, rate limit) degrades silently to an empty list so the rest
of the pipeline (KB + web) carries the report.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

log = logging.getLogger("test_analysis.deep_research")

_TIMEOUT = int(os.getenv("DR_LIT_TIMEOUT", "8"))
_UA = "MediverseAI-DeepResearch/3.0 (occupational health report)"
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# optional NCBI key raises the rate limit but is not required
_NCBI_KEY = os.getenv("NCBI_API_KEY", "")


def _requests():
    import requests
    return requests


# --------------------------------------------------------------------------- #
# PubMed E-utilities: esearch (ids) → esummary (metadata)
# --------------------------------------------------------------------------- #
def _pubmed(query: str, retmax: int) -> List[Dict[str, Any]]:
    requests = _requests()
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json",
              "sort": "relevance"}
    if _NCBI_KEY:
        params["api_key"] = _NCBI_KEY
    r = requests.get(f"{_EUTILS}/esearch.fcgi", params=params,
                     headers={"User-Agent": _UA}, timeout=_TIMEOUT)
    r.raise_for_status()
    ids = (r.json().get("esearchresult", {}) or {}).get("idlist", []) or []
    if not ids:
        return []
    sp = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    if _NCBI_KEY:
        sp["api_key"] = _NCBI_KEY
    s = requests.get(f"{_EUTILS}/esummary.fcgi", params=sp,
                     headers={"User-Agent": _UA}, timeout=_TIMEOUT)
    s.raise_for_status()
    res = s.json().get("result", {}) or {}
    out: List[Dict[str, Any]] = []
    for pmid in ids:
        d = res.get(pmid)
        if not isinstance(d, dict):
            continue
        year = (d.get("pubdate", "") or "")[:4]
        out.append({
            "title": (d.get("title") or "").strip().rstrip("."),
            "journal": d.get("fulljournalname") or d.get("source") or "",
            "year": year,
            "pmid": pmid,
            "doi": next((x.get("value") for x in (d.get("articleids") or [])
                         if x.get("idtype") == "doi"), ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "snippet": "",
            "provider": "PubMed",
        })
    return out


# --------------------------------------------------------------------------- #
# Europe PMC: single REST call with abstracts
# --------------------------------------------------------------------------- #
def _europepmc(query: str, retmax: int) -> List[Dict[str, Any]]:
    requests = _requests()
    params = {"query": query, "format": "json", "pageSize": retmax,
              "resultType": "core"}
    r = requests.get(_EPMC, params=params, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
    r.raise_for_status()
    results = (r.json().get("resultList", {}) or {}).get("result", []) or []
    out: List[Dict[str, Any]] = []
    for d in results:
        doi = d.get("doi", "")
        pmid = d.get("pmid", "")
        url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid
               else (f"https://doi.org/{doi}" if doi else d.get("fullTextUrlList", {}) and ""))
        out.append({
            "title": (d.get("title") or "").strip().rstrip("."),
            "journal": d.get("journalTitle") or d.get("bookOrReportDetails", {}).get("publisher", "") or "",
            "year": str(d.get("pubYear") or ""),
            "pmid": pmid,
            "doi": doi,
            "url": url or "https://europepmc.org/",
            "snippet": (d.get("abstractText") or "")[:600],
            "provider": "Europe PMC",
        })
    return out


def lit_search(query: str, retmax: int = 3) -> List[Dict[str, Any]]:
    """Return up to ``retmax`` biomedical references for ``query``.

    Prefers Europe PMC (ships abstracts in one call); falls back to PubMed. Both
    are best-effort — returns [] on any error.
    """
    if not query:
        return []
    for fn, name in ((_europepmc, "EuropePMC"), (_pubmed, "PubMed")):
        try:
            docs = fn(query, retmax)
            if docs:
                log.info("DR | lit_search(%s) → %d ref(s) via %s", query[:48], len(docs), name)
                return docs[:retmax]
        except Exception as e:  # noqa: BLE001
            log.info("DR | lit_search %s failed (%s)", name, e)
    return []
