# double_rop/chat_service.py
"""
KC (keratoconus / corneal-ectasia) chat assistant.

This is the ROP assistant's pipeline retargeted to keratoconus: same features
(streaming stages, inline citations, automatic web search with an LLM-planned
query, ChatGPT-style document attachments, speech-to-text) but grounded in the
KC knowledge base and answering as a corneal specialist.

The heavy, domain-agnostic machinery (embeddings, web search + scraping +
ranking, source assembly, attachments, transcription, intent detection) is
reused from ``single_rop.chat_service`` so there is a single implementation; only
the knowledge base, the system prompt and the search planner are KC-specific.
"""
from __future__ import annotations

import os
import time
import threading
import logging
from typing import List, Dict, Any, Iterator, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

from doctors_marketplace.services.rag import (
    get_embeddings, chunk_text, read_any_text, normalize_text,
)
from doctors_marketplace.services.llm import LLMClient, _env

# Reuse the domain-agnostic helpers from the ROP service.
from single_rop.chat_service import (
    classify_intent,
    _serper_search, _domain, rank_web_pages,
    build_sources_and_context,
    add_attachment, retrieve_attachments,          # shared uuid-keyed attachment store
    transcribe_audio, RateLimited,                 # shared Whisper STT
    WEB_MAX_PAGES, ENABLE_WEB, _READABLE_SUFFIXES,
)

log = logging.getLogger("kc.chat")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = _env("KC_KB_DIR", default=os.path.join(BASE_DIR, "rag_api", "kc_documents"))
KB_INDEX_DIR = _env("KC_KB_INDEX_DIR",
                    default=os.path.join(BASE_DIR, "vectorstores", "kc_kb"))

# --------------------------------------------------------------------------- #
# Knowledge-base index (built once, at startup) — KC-scoped globals
# --------------------------------------------------------------------------- #
_index_lock = threading.Lock()
_kb_index: Optional[FAISS] = None
_kb_ready = False
_kb_building = False


def _iter_kb_files(kb_dir: str) -> List[str]:
    if not os.path.isdir(kb_dir):
        return []
    files: List[str] = []
    for root, _dirs, names in os.walk(kb_dir):
        for name in names:
            if name.lower().endswith(_READABLE_SUFFIXES):
                files.append(os.path.join(root, name))
    return sorted(files)


def _index_is_fresh(kb_files: List[str]) -> bool:
    faiss_file = os.path.join(KB_INDEX_DIR, "index.faiss")
    if not os.path.isfile(faiss_file):
        return False
    newest = max((os.path.getmtime(p) for p in kb_files), default=0.0)
    return os.path.getmtime(faiss_file) >= newest


def build_or_load_index() -> Optional[FAISS]:
    global _kb_index, _kb_ready
    kb_files = _iter_kb_files(KB_DIR)
    if not kb_files:
        log.warning("KB | no source files under %s — KC assistant runs on web + "
                    "uploads only until you add documents there.", KB_DIR)
        _kb_ready = True
        return None

    embeddings = get_embeddings()
    if _index_is_fresh(kb_files):
        try:
            t0 = time.perf_counter()
            _kb_index = FAISS.load_local(
                KB_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            log.info("KB | loaded persisted index (%d files) in %.2fs",
                     len(kb_files), time.perf_counter() - t0)
            _kb_ready = True
            return _kb_index
        except Exception as e:
            log.warning("KB | persisted index unusable (%s) — rebuilding.", e)

    log.info("KB | building KC index from %d file(s)…", len(kb_files))
    t0 = time.perf_counter()
    docs: List[Document] = []
    for path in kb_files:
        try:
            text = read_any_text(path)
        except Exception as e:
            log.warning("KB | failed to read %s: %s", os.path.basename(path), e)
            continue
        if text.strip():
            docs.extend(chunk_text(text, metadata={"source": os.path.basename(path)}))
    if not docs:
        log.warning("KB | no usable chunks — running without KB.")
        _kb_ready = True
        return None

    _kb_index = FAISS.from_documents(docs, embedding=embeddings)
    os.makedirs(KB_INDEX_DIR, exist_ok=True)
    _kb_index.save_local(KB_INDEX_DIR)
    log.info("KB | ✅ indexed %d chunks from %d file(s) in %.2fs",
             len(docs), len(kb_files), time.perf_counter() - t0)
    _kb_ready = True
    return _kb_index


def _background_build():
    global _kb_building
    with _index_lock:
        if _kb_ready or _kb_building:
            return
        _kb_building = True
    try:
        build_or_load_index()
    except Exception:
        log.exception("KB | indexing failed")
    finally:
        _kb_building = False


def start_background_index():
    log.info("KB | scheduling one-time startup indexing (dir=%s)", KB_DIR)
    threading.Thread(target=_background_build, name="kc-kb-index", daemon=True).start()


def retrieve_local(query: str, k: int = 5) -> List[Document]:
    if _kb_index is None:
        return []
    try:
        t0 = time.perf_counter()
        hits = _kb_index.similarity_search(normalize_text(query), k=k)
        log.info("LOCAL | retrieved %d chunk(s) in %.2fs", len(hits),
                 time.perf_counter() - t0)
        for i, d in enumerate(hits, 1):
            log.info("LOCAL |   [%d] %s :: %s", i, d.metadata.get("source", "N/A"),
                     d.page_content[:120].replace("\n", " "))
        return hits
    except Exception as e:
        log.warning("LOCAL | retrieval failed: %s", e)
        return []


# --------------------------------------------------------------------------- #
# Attachments (reuse the shared store, but expose under this module too)
# --------------------------------------------------------------------------- #
def add_kc_attachment(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    return add_attachment(file_bytes, filename)


# --------------------------------------------------------------------------- #
# LLM + prompt (KC-specific)
# --------------------------------------------------------------------------- #
_llm_singleton: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton


_SYSTEM_PROMPT = """\
You are a warm, clear, and empathetic corneal specialist focused on keratoconus
and corneal ectasia. Behave like a real clinician while strictly controlling when
you use medical content. Mirror the user's language and level of detail.

INPUTS
- [DIAGNOSTIC CONTEXT]: this screening's AI findings — per-eye class (e.g. Normal,
  ATN, NEIr, EIr, eKCN where eKCN = early keratoconus) and a Z classification
  (SfRS / NSfRS). Interpret these as an automated screening result, not a
  definitive diagnosis.
- [SOURCES]: numbered snippets. "uploaded file:" is a document the USER attached
  to this conversation — when present, it is the primary thing to answer about,
  and you must use it fully. "document:" is the clinic library; "web:" is the web.
  Treat these as the ONLY external facts.
- [CHAT HISTORY]: prior messages.
- [USER QUESTION]: the current user message.

CITATIONS (mandatory — the user must see which source each claim came from)
- End EVERY sentence that states a medical fact, number, guideline, or any
  web/document-derived detail with the marker of the exact source it came from,
  e.g. "Corneal cross-linking can halt progression in early keratoconus [2]."
- Attribute precisely: put [n] next to the specific claim it supports — do NOT
  lump all citations at the end. Different claims from different sources carry
  different markers; combine as [2][4] when needed.
- Use only the numbers shown in [SOURCES]; never invent one; only cite what you used.
- Do NOT write your own "References"/"Sources" list — the app renders them.
- Greetings, farewells and small talk carry no citations.

ABSOLUTE RULES
1) Never invent facts. Use medical content only from [DIAGNOSTIC CONTEXT] and/or
   [SOURCES]. If something is missing, say so briefly and suggest an eye-care
   professional / corneal specialist.
2) Tone: concise, reassuring, plain language first; brief clinical terms second.
3) Keep formatting simple Markdown.

INTENT HANDLING (IN PRIORITY ORDER)
A. Greeting: warm 1–2 sentence greeting and invite a question. No medical content.
B. Farewell / thanks: brief, kind reply. No citations.
C. Section question (left eye / right eye / Z class / severity / treatment / plan):
   answer that ONE topic only, ~120–150 words, with citations.
D. General keratoconus question: short integrated explanation using "# Summary"
   then "# Next Steps" bullets, with citations.
E. Other small talk: 1–2 sentences, no medical content unless asked.
"""


def _build_messages(query: str, context_str: str, chat_history: str,
                    diagnostic_context_text: str) -> List[Dict[str, str]]:
    diag_block = (
        f"AI Screening Results:\n---\n{diagnostic_context_text}\n---"
        if diagnostic_context_text
        else "No screening results provided for this query."
    )
    user_block = (
        f"[DIAGNOSTIC CONTEXT]:\n{diag_block}\n\n"
        f"[SOURCES]:\n{context_str or '(no external sources retrieved)'}\n\n"
        f"[CHAT HISTORY]:\n{chat_history or '(none)'}\n\n"
        f"[USER QUESTION]:\n{query}\n\n"
        "Respond per your intent rules, citing sources inline as [n]."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


def plan_web(query: str, chat_history: str, diagnostic_context_text: str,
             local_docs: List[Document]) -> tuple[bool, str]:
    """Decide whether to web-search and craft a self-contained KC query (1 call)."""
    snippet = " ".join(d.page_content[:200] for d in local_docs[:3]) or "(none)"
    try:
        import re, json as _json
        raw = _get_llm().chat([
            {"role": "system", "content":
                "You plan web searches for a keratoconus / corneal-ectasia "
                "assistant. Using the screening findings, the conversation and the "
                "clinic snippets, decide if answering the user's latest message "
                "needs a web search for external or up-to-date information the "
                "snippets do not cover. If so, write ONE concise, self-contained "
                "Google query — resolve pronouns/vague references (\"the surgery\", "
                "\"it\", \"the treatment\") into explicit corneal/keratoconus terms. "
                "Reply with ONLY compact JSON: {\"search\": true|false, \"query\": "
                "\"...\"}. Use search=false for greetings/small talk or when the "
                "snippets already suffice."},
            {"role": "user", "content":
                f"Screening findings: {diagnostic_context_text or '(none)'}\n"
                f"Clinic snippets: {snippet[:1000]}\n"
                f"Conversation so far: {chat_history or '(none)'}\n"
                f"User's latest message: {query}"},
        ], temperature=0).strip()
        m = re.search(r"\{.*\}", raw, re.S)
        data = _json.loads(m.group(0)) if m else {}
        do_search = bool(data.get("search"))
        search_query = (str(data.get("query") or "").strip() or query)
        log.info("PLAN | search=%s query=%r", do_search, search_query)
        return do_search, search_query
    except Exception as e:
        log.warning("PLAN | planning failed (%s) — no web search.", e)
        return False, query


# --------------------------------------------------------------------------- #
# Orchestration — streaming
# --------------------------------------------------------------------------- #
def chat_stream(query: str, diagnostic_context_text: str = "",
                chat_history: str = "", use_web: bool = True,
                attachments: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
    t_start = time.perf_counter()
    intent = classify_intent(query)
    log.info("─" * 70)
    log.info("Q&A | question=%r | intent=%s | use_web=%s", query, intent, use_web)

    yield {"type": "stage", "stage": "thinking", "label": "Thinking"}

    local_docs: List[Document] = []
    web_docs: List[Document] = []
    attach_docs: List[Document] = []

    if intent == "question":
        attach_docs = retrieve_attachments(attachments or [], query)
        local_docs = retrieve_local(query)

        do_search, search_query = (False, query)
        if use_web and ENABLE_WEB and not attach_docs:
            do_search, search_query = plan_web(
                query, chat_history, diagnostic_context_text, local_docs)

        if do_search:
            yield {"type": "stage", "stage": "searching",
                   "label": "Searching the web", "query": search_query}
            organic = _serper_search(search_query)[:WEB_MAX_PAGES]
            previews = [{"title": r.get("title") or _domain(r.get("link", "")),
                         "url": r.get("link"), "domain": _domain(r.get("link", ""))}
                        for r in organic if r.get("link")]
            if previews:
                yield {"type": "stage", "stage": "searching",
                       "label": f"Reading {len(previews)} web source"
                                f"{'s' if len(previews) != 1 else ''}",
                       "sources": previews}
            web_docs = rank_web_pages(search_query, organic)
            yield {"type": "stage", "stage": "thinking", "label": "Thinking"}
    else:
        log.info("Q&A | %s detected — skipping retrieval for a fast reply.", intent)

    sources, context_str = build_sources_and_context(attach_docs + local_docs, web_docs)
    log.info("Q&A | context: %d attached + %d local + %d web chunk(s), %d source(s)",
             len(attach_docs), len(local_docs), len(web_docs), len(sources))

    messages = _build_messages(query, context_str, chat_history, diagnostic_context_text)

    t0 = time.perf_counter()
    answer_parts: List[str] = []
    try:
        streamed = False
        for piece in _get_llm().chat_stream(messages):
            streamed = True
            answer_parts.append(piece)
            yield {"type": "token", "text": piece}
        if not streamed:
            text = _get_llm().chat(messages)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
    except Exception:
        log.exception("LLM | streaming failed — trying non-streaming fallback")
        try:
            text = _get_llm().chat(messages)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
        except Exception:
            log.exception("LLM | generation failed")
            yield {"type": "error",
                   "message": "I couldn't reach the language model just now. "
                              "Please try again in a moment."}
            return

    answer = "".join(answer_parts)
    log.info("LLM | answer generated in %.2fs", time.perf_counter() - t0)
    log.info("A   | %s", answer.replace("\n", " ")[:400])
    log.info("Q&A | done in %.2fs", time.perf_counter() - t_start)

    yield {"type": "done", "sources": sources,
           "local": len(local_docs) + len(attach_docs), "web": len(web_docs)}


def chat_answer(query: str, diagnostic_context_text: str = "",
                chat_history: str = "", use_web: bool = True,
                attachments: Optional[List[str]] = None) -> Dict[str, Any]:
    answer_parts: List[str] = []
    sources: List[Dict] = []
    error: Optional[str] = None
    for ev in chat_stream(query, diagnostic_context_text, chat_history, use_web, attachments):
        if ev["type"] == "token":
            answer_parts.append(ev["text"])
        elif ev["type"] == "done":
            sources = ev["sources"]
        elif ev["type"] == "error":
            error = ev["message"]
    return {"answer": error or "".join(answer_parts), "sources": sources,
            "combined_context_truncated": ""}
