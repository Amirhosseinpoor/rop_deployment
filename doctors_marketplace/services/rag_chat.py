# doctors_marketplace/services/rag_chat.py
"""
Streaming RAG orchestration for the marketplace doctor assistants.

Yields the same live-progress event shapes as the ROP assistant
(single_rop/chat_service.py) so the shared frontend renderer works identically:

    {"type":"stage", "stage":"thinking"|"searching", "label":..., ["query"|"sources"]}
    {"type":"token", "text":...}
    {"type":"done",  "sources":[{id,type,title,url,domain}, ...]}
    {"type":"error", "message":...}

Retrieval is used ONLY WHEN NEEDED:
  * Greetings / farewells / thanks / small talk short-circuit ALL retrieval and
    web search — one fast LLM call, no embeddings, no planning round-trip.
  * Real questions retrieve the doctor's knowledge base, then a small planner LLM
    call decides whether a live web search is also warranted.

Every step logs to the terminal (logger inherits the ``doctors_marketplace``
console handler) so the whole chat pipeline is visible while chatting.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Iterator, List, Union

from langchain_community.docstore.document import Document

from .rag import retrieve_context
from .llm import LLMClient
from . import websearch

log = logging.getLogger(__name__)   # doctors_marketplace.services.rag_chat

Content = Union[str, list]

# --------------------------------------------------------------------------- #
# Intent detection (regex, English + Persian) — mirrors the ROP assistant so a
# "hello" never touches the RAG / web stack.
# --------------------------------------------------------------------------- #
_GREETING_RE = re.compile(
    r"^\s*(hi|hii+|hey+|hello+|yo|salam|sallam|salaam|سلام|درود|hola|"
    r"good\s+(morning|afternoon|evening)|چطوری|خوبی|حالت چطوره|سلام علیکم)[\s!.,؟?]*$", re.I)
_FAREWELL_RE = re.compile(
    r"^\s*(bye+|goodbye|see\s+you|good\s*night|khodahafez|khodafez|"
    r"خداحافظ|خدانگهدار|بای|فعلا)[\s!.,؟?]*$", re.I)
_THANKS_RE = re.compile(
    r"^\s*(thanks?|thank\s*you|thx|ty|مرسی|ممنون|تشکر|سپاس|دمت گرم)[\s!.,؟?]*$", re.I)


def classify_intent(query: str) -> str:
    """Return 'greeting' | 'farewell' | 'thanks' | 'question'."""
    q = (query or "").strip()
    if _GREETING_RE.match(q):
        return "greeting"
    if _FAREWELL_RE.match(q):
        return "farewell"
    if _THANKS_RE.match(q):
        return "thanks"
    return "question"


_CITATION_RULES = """\
SOURCES & CITATIONS (mandatory — the user must see which source each claim came from)
- [SOURCES] is a numbered list of snippets: "document:" items come from this
  doctor's own knowledge base; "web:" items were just fetched from the web.
- End EVERY sentence that states a medical fact, number, guideline, dosage or any
  other detail drawn from a source with the marker of the exact source it came
  from, e.g. "Ibuprofen reduces fever within an hour [2]."
- Attribute precisely: put [n] next to the specific claim it supports — do NOT
  lump all citations at the end. Sentences drawn from different sources must carry
  different markers. If one sentence combines two sources, cite both: [1][3].
- Use only the numbers shown in [SOURCES]; never invent a number, and only cite a
  source you actually used.
- Do NOT write your own "References"/"Sources" list — the app renders source cards
  from your [n] markers automatically.
- Greetings, farewells and pure small talk carry no citations.
- Mirror the user's language (Persian or English) and keep formatting simple Markdown.
"""


def _plan_web(llm: LLMClient, query: str, chat_history: str,
              local_docs: List[Document]) -> tuple[bool, str]:
    """One small LLM call: decide whether a web search is needed and craft a
    self-contained query. Returns (should_search, search_query)."""
    snippet = " ".join(d.page_content[:200] for d in local_docs[:3]) or "(none)"
    t0 = time.perf_counter()
    try:
        raw = llm.chat([
            {"role": "system", "content":
                "You plan web searches for a medical AI assistant. Using the "
                "conversation and the clinic snippets, decide whether answering "
                "the user's latest message needs external or up-to-date medical "
                "information the snippets do not cover. If so, write ONE concise, "
                "fully self-contained web search query — resolve pronouns and vague "
                "references (\"it\", \"the treatment\", \"that drug\") into explicit "
                "terms. Reply with ONLY compact JSON: {\"search\": true|false, "
                "\"query\": \"...\"}. Use search=false for greetings, small talk, or "
                "when the clinic snippets already suffice."},
            {"role": "user", "content":
                f"Clinic snippets: {snippet[:1000]}\n"
                f"Conversation so far: {chat_history or '(none)'}\n"
                f"User's latest message: {query}"},
        ], temperature=0).strip()
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        do_search = bool(data.get("search"))
        search_query = (str(data.get("query") or "").strip() or query)
        log.info("PLAN | search=%s query=%r (%.2fs)", do_search, search_query,
                 time.perf_counter() - t0)
        return do_search, search_query
    except Exception as e:
        log.warning("PLAN | planning failed (%s) — no web search.", e)
        return False, query


def stream_doctor_answer(
    *,
    doctor,
    query: str,
    system_text: str,
    history_msgs: List[Dict[str, Content]],
    user_content: Content,
    use_web: bool = True,
    has_doc_attachments: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Run the retrieval + generation pipeline, yielding live events."""
    llm = LLMClient()
    t_start = time.perf_counter()
    intent = classify_intent(query)
    log.info("─" * 70)
    log.info("Q&A | doctor=%s | intent=%s | use_web=%s | attach=%s | q=%r",
             getattr(doctor, "slug", doctor), intent, use_web,
             has_doc_attachments, (query or "")[:140])

    yield {"type": "stage", "stage": "thinking", "label": "Thinking"}

    local_docs: List[Document] = []
    web_docs: List[Document] = []

    if intent == "question":
        # 1) Doctor's own knowledge base.
        t0 = time.perf_counter()
        try:
            local_docs = retrieve_context(doctor, query or "", k=5)
        except Exception as e:
            log.warning("LOCAL | retrieval failed: %s", e)
        log.info("LOCAL | retrieved %d KB chunk(s) in %.2fs",
                 len(local_docs), time.perf_counter() - t0)
        for i, d in enumerate(local_docs, 1):
            log.info("LOCAL |   [%d] %s :: %s", i,
                     (d.metadata or {}).get("title", "KB"),
                     (d.page_content or "")[:100].replace("\n", " "))

        # 2) Decide + run a live web search only if warranted.
        hist_str = "\n".join(
            f"{m['role']}: {m['content'] if isinstance(m['content'], str) else '(attachment)'}"
            for m in history_msgs[-6:]
        )
        if use_web and websearch.ENABLE_WEB and query and not has_doc_attachments:
            do_search, search_query = _plan_web(llm, query, hist_str, local_docs)
            if do_search:
                yield {"type": "stage", "stage": "searching",
                       "label": "Searching the web", "query": search_query}
                organic = websearch.serper_search(search_query)[:websearch.WEB_MAX_PAGES]
                previews = [{"title": r.get("title") or websearch.domain_of(r.get("link", "")),
                             "url": r.get("link"),
                             "domain": websearch.domain_of(r.get("link", ""))}
                            for r in organic if r.get("link")]
                for i, p in enumerate(previews, 1):
                    log.info("WEB |   [%d] %s — %s", i, p["title"][:60], p["url"])
                if previews:
                    n = len(previews)
                    yield {"type": "stage", "stage": "searching",
                           "label": f"Reading {n} web source{'s' if n != 1 else ''}",
                           "sources": previews}
                web_docs = websearch.rank_web_pages(search_query, organic)
                yield {"type": "stage", "stage": "thinking", "label": "Thinking"}
        else:
            log.info("WEB | skipped (use_web=%s enabled=%s attach=%s)",
                     use_web, websearch.ENABLE_WEB, has_doc_attachments)
    else:
        log.info("Q&A | %s — skipping KB retrieval + web search for a fast reply.", intent)

    sources, context_str = websearch.build_sources_and_context(local_docs, web_docs)
    log.info("Q&A | context: %d local + %d web chunk(s) -> %d source(s), %d ctx chars",
             len(local_docs), len(web_docs), len(sources), len(context_str))

    # Assemble the OpenAI-format messages. Small talk gets a lean system prompt
    # (no citation rules / sources) so a greeting is a single small LLM call.
    if intent == "question":
        system_msg = (system_text or "") + "\n\n" + _CITATION_RULES
    else:
        system_msg = (system_text or "") + (
            "\n\nThe user is making small talk (a greeting, farewell or thanks). "
            "Reply warmly and briefly in 1–2 sentences, in the user's language, "
            "with no medical content and no citations.")
    system_msg += ("\n\nWrite any math or formulas in plain text/Unicode (e.g. \"BMI = 26.1 kg/m²\"); "
                   "do NOT use LaTeX, dollar signs, or backslash commands.")
    messages: List[Dict[str, Content]] = [{"role": "system", "content": system_msg}]
    if context_str:
        messages.append({"role": "system", "content": "[SOURCES]:\n" + context_str})
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_content})

    log.info("LLM | generating (model=%s, %d msgs, %d history turns)…",
             llm.model, len(messages), len(history_msgs))
    t_llm = time.perf_counter()
    answer_parts: List[str] = []
    try:
        streamed = False
        for piece in llm.chat_stream(messages):
            streamed = True
            answer_parts.append(piece)
            yield {"type": "token", "text": piece}
        if not streamed:
            log.info("LLM | stream empty — non-streaming fallback")
            text = llm.chat(messages)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
    except Exception:
        log.exception("LLM | streaming failed — trying non-streaming fallback")
        try:
            text = llm.chat(messages)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
        except Exception:
            log.exception("LLM | generation failed")
            yield {"type": "error",
                   "message": "متاسفم—الان به مدل پزشکی دسترسی ندارم. لطفاً کمی بعد دوباره تلاش کنید."}
            return

    answer = "".join(answer_parts).strip()
    log.info("LLM | answer %d chars in %.2fs", len(answer), time.perf_counter() - t_llm)
    log.info("A   | %s", answer.replace("\n", " ")[:300])
    log.info("Q&A | done in %.2fs (%d source(s))", time.perf_counter() - t_start, len(sources))

    yield {"type": "done", "sources": sources, "answer": answer}
