# single_rop/chat_service.py
"""
Self-contained RAG chat backend for the ROP assistant.

Design goals (why this module exists)
-------------------------------------
* Runs fully in-process under `python manage.py runserver` — no FastAPI /
  uvicorn, no Celery, no Redis.
* Embeddings via GAPGPT `text-embedding-3-large` (OpenAI-compatible), reusing
  ``doctors_marketplace.services.rag`` — no local model, no torch, no Ollama.
* Generation via ``doctors_marketplace.services.llm.LLMClient`` (OpenAI SDK with
  retries + streaming).
* The local knowledge base is indexed **once, at startup** (background thread),
  and the FAISS store is persisted — never rebuilt during a Q&A.
* Greetings / farewells / small talk short-circuit all retrieval, so "hello" is
  a single fast LLM call instead of a full web-RAG round trip.
* ``chat_stream`` yields staged events (thinking → searching → reading →
  writing) and streams answer tokens, so the UI can show live progress and
  ChatGPT-style inline source citations.
"""
from __future__ import annotations

import os
import re
import time
import uuid
import tempfile
import threading
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Iterator, Optional

import requests
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

# Reuse the marketplace's GAPGPT plumbing (no torch / Ollama).
from doctors_marketplace.services.rag import (
    get_embeddings,
    chunk_text,
    read_any_text,
    normalize_text,
)
from doctors_marketplace.services.llm import LLMClient, _env

log = logging.getLogger("rop.chat")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KB_DIR = _env("ROP_KB_DIR",
              default=os.path.join(BASE_DIR, "rag_api", "rag_documents"))
KB_INDEX_DIR = _env("ROP_KB_INDEX_DIR",
                    default=os.path.join(BASE_DIR, "vectorstores", "rop_kb"))

SERPER_API_KEY = _env("SERPER_API_KEY",
                      default="3fd967e3d342dd9e6641649e13850ed8ed2d1ee5")
SERPER_API_URL = _env("SERPER_API_URL", default="https://google.serper.dev/search")

ENABLE_WEB = _env("ROP_ENABLE_WEB", default="1") not in ("0", "false", "False", "no")
WEB_MAX_PAGES = int(_env("ROP_WEB_MAX_PAGES", default="5"))  # top-N websites to read
STT_MODEL = _env("GAPGPT_STT_MODEL", "OPENAI_STT_MODEL", default="whisper-1")

_READABLE_SUFFIXES = (".pdf", ".txt", ".md", ".docx", ".doc")

# --------------------------------------------------------------------------- #
# Knowledge-base index (built once, at startup)
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


def _newest_mtime(paths: List[str]) -> float:
    return max((os.path.getmtime(p) for p in paths), default=0.0)


def _index_is_fresh(kb_files: List[str]) -> bool:
    faiss_file = os.path.join(KB_INDEX_DIR, "index.faiss")
    if not os.path.isfile(faiss_file):
        return False
    return os.path.getmtime(faiss_file) >= _newest_mtime(kb_files)


def _load_kb_documents(kb_files: List[str]) -> List[Document]:
    docs: List[Document] = []
    for path in kb_files:
        try:
            text = read_any_text(path)
        except Exception as e:
            log.warning("KB | failed to read %s: %s", os.path.basename(path), e)
            continue
        if not text.strip():
            continue
        chunks = chunk_text(text, metadata={"source": os.path.basename(path)})
        log.info("KB | %-40s -> %d chunks", os.path.basename(path), len(chunks))
        docs.extend(chunks)
    return docs


def build_or_load_index() -> Optional[FAISS]:
    """Build the KB FAISS index, or load the persisted one if it is fresh."""
    global _kb_index, _kb_ready

    kb_files = _iter_kb_files(KB_DIR)
    if not kb_files:
        log.warning("KB | no source files under %s — assistant will run without "
                    "local knowledge base.", KB_DIR)
        _kb_ready = True
        return None

    embeddings = get_embeddings()

    if _index_is_fresh(kb_files):
        try:
            t0 = time.perf_counter()
            _kb_index = FAISS.load_local(
                KB_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            log.info("KB | loaded persisted index from %s (%d files) in %.2fs",
                     KB_INDEX_DIR, len(kb_files), time.perf_counter() - t0)
            _kb_ready = True
            return _kb_index
        except Exception as e:
            log.warning("KB | persisted index unusable (%s) — rebuilding.", e)

    log.info("KB | building index from %d source file(s)…", len(kb_files))
    t0 = time.perf_counter()
    docs = _load_kb_documents(kb_files)
    if not docs:
        log.warning("KB | no usable chunks extracted — running without KB.")
        _kb_ready = True
        return None

    _kb_index = FAISS.from_documents(docs, embedding=embeddings)
    os.makedirs(KB_INDEX_DIR, exist_ok=True)
    _kb_index.save_local(KB_INDEX_DIR)
    log.info("KB | ✅ indexed %d chunks from %d file(s) in %.2fs (saved to %s)",
             len(docs), len(kb_files), time.perf_counter() - t0, KB_INDEX_DIR)
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
    """Kick off KB indexing in a daemon thread so `runserver` starts instantly."""
    log.info("KB | scheduling one-time startup indexing (dir=%s)", KB_DIR)
    threading.Thread(target=_background_build, name="rop-kb-index",
                     daemon=True).start()


# --------------------------------------------------------------------------- #
# User-attached documents (ChatGPT-style)
# --------------------------------------------------------------------------- #
# An uploaded file is a *per-conversation* attachment the assistant reads to
# answer questions about it — it is NEVER added to the shared clinic knowledge
# base. Attachments live in memory only (ephemeral, like a chat session).
#   * Small docs are kept whole and stuffed into the prompt (full awareness).
#   * Larger docs are embedded into their own FAISS store and retrieved per query.
_ATTACH_FULLTEXT_LIMIT = 9000          # chars: keep whole doc if under this
_ATTACH_MAX_DOCS = 64                  # cap the in-memory store
_attach_lock = threading.Lock()
_attachments: "Dict[str, Dict[str, Any]]" = {}


def add_attachment(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Register an uploaded document as a conversation attachment (not KB). Returns
    {"doc_id", "title", "filename", "chunks"}. Raises ValueError if unreadable.
    """
    safe = os.path.basename(filename or "document")
    if not safe.lower().endswith(_READABLE_SUFFIXES):
        raise ValueError("Unsupported file type. Upload a PDF, DOCX, TXT or MD file.")

    suffix = os.path.splitext(safe)[1] or ".txt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(file_bytes)
        tmp.close()
        text = read_any_text(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    if not text.strip():
        raise ValueError("No readable text found in the document.")
    docs = chunk_text(text, metadata={"source": safe, "origin": "attachment"})
    if not docs:
        raise ValueError("The document produced no usable text chunks.")

    doc_id = uuid.uuid4().hex
    title = _pretty_title(safe)
    full_text = text if len(text) <= _ATTACH_FULLTEXT_LIMIT else None
    index = None
    if full_text is None:                       # large doc -> per-doc FAISS
        index = FAISS.from_documents(docs, embedding=get_embeddings())

    with _attach_lock:
        if len(_attachments) >= _ATTACH_MAX_DOCS:      # evict oldest
            _attachments.pop(next(iter(_attachments)))
        _attachments[doc_id] = {
            "title": title, "filename": safe,
            "chunks": docs, "index": index, "full_text": full_text,
        }

    log.info("DOC | attached %r as %s (%d chunks, %s)", safe, doc_id[:8],
             len(docs), "whole" if full_text else "indexed")
    return {"doc_id": doc_id, "title": title, "filename": safe, "chunks": len(docs)}


def retrieve_attachments(attachment_ids: List[str], query: str,
                         per_doc_k: int = 8) -> List[Document]:
    """Gather relevant content from the conversation's attached documents."""
    if not attachment_ids:
        return []
    with _attach_lock:
        entries = [(i, _attachments.get(i)) for i in attachment_ids]

    out: List[Document] = []
    for doc_id, entry in entries:
        if not entry:
            log.info("DOC | attachment %s not found (expired?)", str(doc_id)[:8])
            continue
        meta = {"source": entry["filename"], "title": entry["title"],
                "origin": "attachment"}
        if entry["full_text"] is not None:
            out.append(Document(page_content=entry["full_text"], metadata=dict(meta)))
        else:
            try:
                hits = entry["index"].similarity_search(normalize_text(query), k=per_doc_k)
            except Exception as e:
                log.warning("DOC | retrieval failed for %s: %s", entry["filename"], e)
                hits = []
            for h in hits:
                h.metadata.update(meta)
                out.append(h)
    if out:
        log.info("DOC | attachments contributed %d passage(s)", len(out))
    return out


class RateLimited(Exception):
    """Raised when the transcription provider is rate-limited/over quota (HTTP 429)."""


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm",
                     max_retries: int = 3) -> Dict[str, str]:
    """
    Transcribe speech to text with automatic language detection (Whisper handles
    Persian and English without being told which). Returns {"text", "language"}.

    Transient rate limits (HTTP 429) are retried with a short backoff; if they
    persist, a RateLimited error is raised so the UI can show a friendly notice.
    """
    from openai import OpenAI
    from doctors_marketplace.services.llm import API_KEY, BASE_URL
    import io

    if not API_KEY:
        raise RuntimeError("No API key for transcription.")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    name = filename or "audio.webm"

    last_err = None
    for attempt in range(max_retries):
        bio = io.BytesIO(audio_bytes)      # fresh handle each attempt
        bio.name = name
        try:
            resp = client.audio.transcriptions.create(
                model=STT_MODEL, file=bio, response_format="verbose_json")
            text = (getattr(resp, "text", "") or "").strip()
            language = getattr(resp, "language", "") or ""
            log.info("STT | %d bytes -> lang=%s text=%r", len(audio_bytes),
                     language, text[:80])
            return {"text": text, "language": language}
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            status = getattr(e, "status_code", None)
            rate_limited = (status == 429 or "429" in msg or "api_limit" in msg
                            or "rate" in msg or "quota" in msg)
            transient = rate_limited or status in (500, 502, 503)
            if transient and attempt < max_retries - 1:
                wait = 1.5 * (attempt + 1)
                log.warning("STT | transient error (%s) — retrying in %.1fs", status, wait)
                time.sleep(wait)
                continue
            if rate_limited:
                log.warning("STT | rate limited: %s", str(e)[:200])
                raise RateLimited(str(e))
            raise
    raise last_err  # pragma: no cover


# --------------------------------------------------------------------------- #
# Intent detection
# --------------------------------------------------------------------------- #
_GREETING_RE = re.compile(
    r"^\s*(hi|hii|hey+|hello+|yo|salam|sallam|salaam|سلام|درود|hola|"
    r"good\s+(morning|afternoon|evening)|چطوری|خوبی)[\s!.,؟?]*$", re.I)
_FAREWELL_RE = re.compile(
    r"^\s*(bye+|goodbye|see\s+you|good\s*night|khodahafez|khodafez|"
    r"خداحافظ|خدانگهدار|بای)[\s!.,؟?]*$", re.I)
_THANKS_RE = re.compile(
    r"^\s*(thanks?|thank\s*you|thx|ty|مرسی|ممنون|تشکر|سپاس)[\s!.,؟?]*$", re.I)


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


# --------------------------------------------------------------------------- #
# Local retrieval
# --------------------------------------------------------------------------- #
def retrieve_local(query: str, k: int = 5) -> List[Document]:
    if _kb_index is None:
        if not _kb_ready:
            log.info("LOCAL | index not ready yet — skipping local retrieval.")
        return []
    try:
        t0 = time.perf_counter()
        # Plain similarity (not MMR): the KB is a decision table of near-identical
        # "Zone/Stage/Plus -> Decision" rows, where MMR's diversity would push the
        # exact matching row out of the top-k.
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
# Web retrieval (Serper -> scrape top N -> ephemeral FAISS)
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _serper_search(query: str) -> List[Dict[str, Any]]:
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
        for i, r in enumerate(organic[:WEB_MAX_PAGES], 1):
            log.info("WEB |   [%d] %s — %s", i, (r.get("title") or "")[:70],
                     r.get("link", ""))
        return organic
    except Exception as e:
        log.warning("WEB | Serper search failed: %s", e)
        return []


def _scrape_url(url: str, char_limit: int = 16000) -> str:
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


def plan_web(query: str, chat_history: str, diagnostic_context_text: str,
             local_docs: List[Document]) -> tuple[bool, str]:
    """
    In ONE small LLM call, decide whether to search the web and, if so, craft a
    self-contained search query. The model resolves pronouns and ambiguity
    ("the surgery", "it", "how long") using the conversation and the diagnostic
    findings — e.g. a bare "how long does treatment take after surgery?" becomes
    "how long does treatment take after ROP surgery?".

    Returns (should_search, search_query).
    """
    snippet = " ".join(d.page_content[:200] for d in local_docs[:3]) or "(none)"
    try:
        raw = _get_llm().chat([
            {"role": "system", "content":
                "You plan web searches for a pediatric ROP (retinopathy of "
                "prematurity) assistant. Using the diagnostic findings, the "
                "conversation, and the clinic snippets, decide if answering the "
                "user's latest message needs a web search for external or "
                "up-to-date information the snippets do not cover. If so, write ONE "
                "concise, fully self-contained Google query — resolve pronouns and "
                "vague references (\"the surgery\", \"it\", \"the treatment\") into "
                "explicit ROP terms. Reply with ONLY compact JSON: "
                "{\"search\": true|false, \"query\": \"...\"}. Use search=false for "
                "greetings, small talk, or when the clinic snippets already suffice."},
            {"role": "user", "content":
                f"Diagnostic findings: {diagnostic_context_text or '(none)'}\n"
                f"Clinic snippets: {snippet[:1000]}\n"
                f"Conversation so far: {chat_history or '(none)'}\n"
                f"User's latest message: {query}"},
        ], temperature=0).strip()

        import json as _json
        m = re.search(r"\{.*\}", raw, re.S)
        data = _json.loads(m.group(0)) if m else {}
        do_search = bool(data.get("search"))
        search_query = (str(data.get("query") or "").strip() or query)
        log.info("PLAN | search=%s query=%r", do_search, search_query)
        return do_search, search_query
    except Exception as e:
        log.warning("PLAN | planning failed (%s) — no web search.", e)
        return False, query


def rank_web_pages(query: str, organic: List[Dict[str, Any]],
                   k: int = 6) -> List[Document]:
    """Scrape the given result pages, embed and similarity-rank their chunks."""
    docs: List[Document] = []
    for r in organic:
        url = r.get("link")
        if not url:
            continue
        title = r.get("title") or _domain(url)
        text = _scrape_url(url)
        if not text:
            continue
        docs.extend(chunk_text(
            text, metadata={"source": url, "origin": "web", "title": title}))

    if not docs:
        log.info("WEB | no usable web content extracted.")
        return []

    try:
        t0 = time.perf_counter()
        vs = FAISS.from_documents(docs, embedding=get_embeddings())
        hits = vs.similarity_search(query, k=k)
        log.info("WEB | ranked %d/%d web chunk(s) in %.2fs", len(hits), len(docs),
                 time.perf_counter() - t0)
        for i, d in enumerate(hits, 1):
            log.info("WEB |   [%d] %s :: %s", i, d.metadata.get("source", ""),
                     d.page_content[:120].replace("\n", " "))
        return hits
    except Exception as e:
        log.warning("WEB | embedding/ranking failed: %s", e)
        return []


# --------------------------------------------------------------------------- #
# Sources + context assembly (numbered for inline [n] citations)
# --------------------------------------------------------------------------- #
def _pretty_title(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename or "document"))[0]
    return re.sub(r"[_\-]+", " ", stem).strip() or "document"


def build_sources_and_context(local_docs: List[Document],
                              web_docs: List[Document]) -> tuple[List[Dict], str]:
    """
    Assign a stable citation id to each distinct source and build the context
    string the LLM sees, each chunk prefixed with its [id] marker.
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
        title = d.metadata.get("title") or _pretty_title(d.metadata.get("source", "document"))
        is_attach = (d.metadata or {}).get("origin") == "attachment"
        kind = "uploaded file" if is_attach else "document"
        sid = _source_id(("attach::" if is_attach else "local::") + title,
                         {"type": "local", "title": title, "url": None,
                          "domain": None, "attachment": is_attach})
        lines.append(f"[{sid}] ({kind}: {title})\n{d.page_content}")

    for d in web_docs:
        url = d.metadata.get("source", "")
        title = d.metadata.get("title") or _domain(url)
        sid = _source_id("web::" + url,
                         {"type": "web", "title": title, "url": url,
                          "domain": _domain(url)})
        lines.append(f"[{sid}] (web: {title} — {url})\n{d.page_content}")

    context_str = "\n---\n".join(lines)
    MAX_CTX = 14000
    if len(context_str) > MAX_CTX:
        context_str = context_str[:MAX_CTX] + "\n[...truncated retrieval context...]"
    return sources, context_str


# --------------------------------------------------------------------------- #
# Prompt + generation
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """\
You are a warm, clear, and empathetic pediatric retina specialist (ROP). Behave
like a real clinician in conversation while strictly controlling when you use
medical content. Mirror the user's language and level of detail.

INPUTS
- [DIAGNOSTIC CONTEXT]: the current findings (Zone, Stage, Plus, Final Decision).
- [SOURCES]: numbered snippets. "uploaded file:" is a document the USER attached to
  this conversation — when present, it is the primary thing to answer about, and
  you must use it fully. "document:" is the clinic library; "web:" is the web.
  Treat these as the ONLY external facts.
- [CHAT HISTORY]: prior messages.
- [USER QUESTION]: the current user message.

CITATIONS (mandatory — the user must see which source each claim came from)
- End EVERY sentence that states a medical fact, number, guideline, or any
  web/document-derived detail with the marker of the exact source it came from,
  e.g. "Anti-VEGF therapy is an option for zone I disease [3]."
- Attribute precisely: put [n] next to the specific claim it supports — do NOT
  lump all citations at the end of the answer. Different sentences drawn from
  different sources must carry different markers.
- If one sentence combines two sources, cite both: [2][4].
- Use only the numbers shown in [SOURCES]; never invent a number, and only cite
  a source you actually used.
- Do NOT write your own "References"/"Sources" list — the app renders the source
  cards from your [n] markers automatically.
- Greetings, farewells and small talk carry no citations.

ABSOLUTE RULES
1) Never invent facts. Use medical content only from [DIAGNOSTIC CONTEXT] and/or
   [SOURCES]. If something is missing, say so briefly and suggest asking the care
   team.
2) Tone: concise, reassuring, plain language first; brief clinical terms second.
3) Keep formatting simple Markdown.

INTENT HANDLING (IN PRIORITY ORDER)
A. Greeting: warm 1–2 sentence greeting and invite a question. No medical content,
   no citations.
B. Farewell / thanks: brief, kind reply. No citations.
C. Section question (zone/stage/plus/final decision/treatment/plan): answer that
   ONE section only, ~120–150 words, with citations.
D. General disease question: short integrated explanation using "# Summary" then
   "# Next Steps" bullets, with citations.
E. Other small talk: 1–2 sentences, no medical content unless asked.
"""


def _build_messages(query: str, context_str: str, chat_history: str,
                    diagnostic_context_text: str) -> List[Dict[str, str]]:
    diag_block = (
        f"Image-Based Diagnostic Results:\n---\n{diagnostic_context_text}\n---"
        if diagnostic_context_text
        else "No Image-Based Diagnostic Results provided for this query."
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


_llm_singleton: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton


# --------------------------------------------------------------------------- #
# Public orchestration — streaming
# --------------------------------------------------------------------------- #
def chat_stream(query: str, diagnostic_context_text: str = "",
                chat_history: str = "", use_web: bool = True,
                attachments: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
    """
    Run the pipeline and yield events for a live UI:
      {"type":"stage",  "stage":..., "label":..., ["query"|"sources"]}
      {"type":"token",  "text":...}
      {"type":"done",   "sources":[...], "local":n, "web":n}
      {"type":"error",  "message":...}
    """
    t_start = time.perf_counter()
    intent = classify_intent(query)
    log.info("─" * 70)
    log.info("Q&A | question=%r | intent=%s | use_web=%s", query, intent, use_web)

    yield {"type": "stage", "stage": "thinking", "label": "Thinking"}

    local_docs: List[Document] = []
    web_docs: List[Document] = []
    attach_docs: List[Document] = []

    if intent == "question":
        # The user's uploaded documents come first — the assistant must be fully
        # aware of them (ChatGPT-style). Then the clinic knowledge base.
        attach_docs = retrieve_attachments(attachments or [], query)
        local_docs = retrieve_local(query)

        # The assistant decides for itself whether to search — and, if so, writes
        # a context-aware query from the history + question + diagnostic results.
        # When the user attached a document, prefer answering from it (no web).
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
            # Back to thinking while the answer is composed.
            yield {"type": "stage", "stage": "thinking", "label": "Thinking"}
    else:
        log.info("Q&A | %s detected — skipping retrieval for a fast reply.", intent)

    # Attachments lead the local context so they are never truncated away.
    sources, context_str = build_sources_and_context(attach_docs + local_docs, web_docs)
    log.info("Q&A | context: %d attached + %d local + %d web chunk(s), %d source(s), %d chars",
             len(attach_docs), len(local_docs), len(web_docs), len(sources), len(context_str))

    messages = _build_messages(query, context_str, chat_history,
                               diagnostic_context_text)

    t0 = time.perf_counter()
    answer_parts: List[str] = []
    try:
        streamed = False
        for piece in _get_llm().chat_stream(messages):
            streamed = True
            answer_parts.append(piece)
            yield {"type": "token", "text": piece}
        if not streamed:  # provider returned nothing streamed — fall back
            text = _get_llm().chat(messages)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
    except Exception:
        log.exception("LLM | streaming failed — trying non-streaming fallback")
        try:
            text = _get_llm().chat(messages)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
        except Exception as e:
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
           "local": len(local_docs), "web": len(web_docs)}


# --------------------------------------------------------------------------- #
# Public orchestration — non-streaming (JSON fallback)
# --------------------------------------------------------------------------- #
def chat_answer(query: str, diagnostic_context_text: str = "",
                chat_history: str = "", use_web: bool = True,
                attachments: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect the streamed pipeline into a single JSON response."""
    answer_parts: List[str] = []
    sources: List[Dict] = []
    error: Optional[str] = None
    for ev in chat_stream(query, diagnostic_context_text, chat_history, use_web,
                          attachments):
        if ev["type"] == "token":
            answer_parts.append(ev["text"])
        elif ev["type"] == "done":
            sources = ev["sources"]
        elif ev["type"] == "error":
            error = ev["message"]
    return {
        "answer": error or "".join(answer_parts),
        "sources": sources,
        "combined_context_truncated": "",
    }
