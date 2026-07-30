# doctors_marketplace/views.py
import base64
import mimetypes
import os
import tempfile

from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST
from django.contrib import messages

from django.db.models import Q

from .models import Doctor, ChatSession, ChatMessage, ChatAttachment, DoctorKnowledge
from .forms import DoctorForm, KnowledgeUploadForm
from .services.llm import LLMClient
from .services.rag import retrieve_context, read_any_text
from .services import rag_chat
from .services import agent_runtime


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DOC_EXTS = {".pdf", ".txt", ".md", ".csv", ".docx", ".doc", ".rtf"}
MAX_IMAGES = 4
MAX_DOC_CHARS = 6000          # per document
MAX_DOC_CHARS_TOTAL = 14000   # across all documents in one turn


def _is_superuser(user):
    return user.is_authenticated and user.is_superuser


# -------- MARKET (public) --------
def market_index(request):
    doctors = Doctor.objects.filter(is_active=True).order_by('name_fa', 'name')
    return render(request, 'doctors_marketplace/market_index2.html', {"doctors": doctors})


def doctor_detail(request, slug):
    doctor = get_object_or_404(Doctor, slug=slug, is_active=True)
    if request.method == 'POST':
        if not request.user.is_authenticated:
            return redirect('account_login')
        # Reuse an existing empty chat instead of piling up blank "New chat"
        # sessions when the button is clicked repeatedly.
        existing = (ChatSession.objects
                    .filter(user=request.user, doctor=doctor)
                    .order_by('-updated_at').first())
        if existing and not existing.messages.exclude(role=ChatMessage.Role.SYSTEM).exists():
            return redirect('doctors_marketplace:chat', session_id=existing.id)
        # Title is generated from the user's first message (ChatGPT/Gemini style).
        session = ChatSession.objects.create(user=request.user, doctor=doctor, title="")
        sys_text = doctor.system_prompt or (
            "شما یک دستیار پزشکی فارسی‌زبان هستید. به احوالپرسی پاسخ کوتاه بدهید و سپس به موضوع تخصص پزشک برگردید."
        )
        ChatMessage.objects.create(session=session, role=ChatMessage.Role.SYSTEM, content=sys_text)
        return redirect('doctors_marketplace:chat', session_id=session.id)
    return render(request, 'doctors_marketplace/doctor_detail2.html', {"doctor": doctor})


# -------- CHAT --------
def _safe_json(obj):
    """JSON string safe to embed in a <script type=application/json> tag."""
    import json
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


@login_required
def chat_view(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages_qs = list(session.messages.select_related('reply_to')
                       .prefetch_related('attachments').all())
    # Give each assistant message a JSON payload the client re-renders with
    # Markdown + inline [n] citations + Sources cards.
    for m in messages_qs:
        if m.role == ChatMessage.Role.ASSISTANT:
            m.payload_json = _safe_json({"raw": m.content or "",
                                         "sources": m.sources or []})
    history = ChatSession.objects.filter(user=request.user, doctor=session.doctor).order_by('-pinned', '-updated_at')[:30]
    return render(request, 'doctors_marketplace/chat.html', {
        "session": session,
        "messages": messages_qs,
        "history": history,
    })


def _classify(upload) -> str:
    ext = os.path.splitext(upload.name)[1].lower()
    if ext in IMAGE_EXTS or (upload.content_type or "").startswith("image/"):
        return ChatAttachment.Kind.IMAGE
    return ChatAttachment.Kind.DOCUMENT


def _image_data_url(attachment: ChatAttachment) -> str | None:
    try:
        with attachment.file.open("rb") as f:
            raw = f.read()
    except Exception:
        return None
    mime = mimetypes.guess_type(attachment.file.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _document_text(attachment: ChatAttachment) -> str:
    """Extract text from a document attachment (best-effort, truncated)."""
    try:
        # FileField may be on remote storage; copy to a temp path if needed.
        path = attachment.file.path
    except Exception:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(attachment.file.name)[1]
        ) as tmp:
            tmp.write(attachment.file.read())
            path = tmp.name
    try:
        text = read_any_text(path).strip()
    except Exception:
        return ""
    return text[:MAX_DOC_CHARS]


def _title_messages(text):
    return [
        {"role": "system", "content": (
            "You generate a very short, descriptive chat title from the user's "
            "first message. Rules: at most 6 words; no quotation marks; no trailing "
            "punctuation; output ONLY the title with nothing else. "
            "CRITICAL — match the language of the user's message: if the user writes "
            "in English, the title MUST be in English; if the user writes in Persian "
            "(Farsi), the title MUST be in Persian. Never translate to another language.\n"
            "یک عنوان بسیار کوتاه و گویا برای گفتگو بساز. حداکثر ۶ کلمه، بدون علامت "
            "نقل‌قول و بدون نقطهٔ پایانی. عنوان را حتماً به همان زبانِ پیام کاربر بنویس؛ "
            "پیام انگلیسی ⇐ عنوان انگلیسی، پیام فارسی ⇐ عنوان فارسی."
        )},
        {"role": "user", "content": (text or "")[:500]},
    ]


def _clean_title(raw):
    return " ".join((raw or "").split()).strip('«»"\'`.:،-  ')[:60]


def _fallback_title(text):
    return " ".join((text or "").split())[:40]


def _generate_title(user_text):
    """A short ChatGPT/Gemini-style conversation title from the first message."""
    text = (user_text or "").strip()
    if not text:
        return ""
    try:
        title = _clean_title(LLMClient().chat(_title_messages(text), temperature=0.3))
        if title:
            return title
    except Exception:
        pass
    return _fallback_title(text)


def _persist_turn(session, user_text, uploads):
    """Save the user message + its attachments; return (user_msg, images, docs)."""
    user_msg = ChatMessage.objects.create(
        session=session, role=ChatMessage.Role.USER, content=user_text)
    images, docs = [], []
    for up in uploads:
        kind = _classify(up)
        att = ChatAttachment.objects.create(
            message=user_msg, kind=kind, file=up, original_name=up.name[:255])
        (images if kind == ChatAttachment.Kind.IMAGE else docs).append(att)
    return user_msg, images[:MAX_IMAGES], docs


def _prepare_turn(session, user_text, image_attachments, doc_attachments, quote=None):
    """Assemble the inputs the streaming RAG orchestrator needs:
    system prompt, prior-turn history, and this turn's (multimodal) content."""
    system = session.messages.filter(role=ChatMessage.Role.SYSTEM).first()
    system_text = system.content if system else (
        session.doctor.system_prompt or "شما یک دستیار پزشکی فارسی‌زبان هستید.")

    history = list(session.messages.exclude(role=ChatMessage.Role.SYSTEM).order_by('created_at'))
    history_msgs = [{"role": m.role, "content": m.content or "(پیوست)"}
                    for m in history[:-1]]

    # Document text -> a compact text block appended to the user's turn.
    doc_blocks, total = [], 0
    for att in doc_attachments:
        txt = _document_text(att)
        if not txt:
            continue
        if total + len(txt) > MAX_DOC_CHARS_TOTAL:
            txt = txt[:max(0, MAX_DOC_CHARS_TOTAL - total)]
        total += len(txt)
        doc_blocks.append(f"### {att.original_name}\n{txt}")
        if total >= MAX_DOC_CHARS_TOTAL:
            break

    turn_text = user_text
    if quote:
        turn_text = f"(در پاسخ به این پیام قبلی):\n> {quote[:500]}\n\n" + (turn_text or "")
    if doc_blocks:
        turn_text = (turn_text + "\n\n" if turn_text else "") + \
            "محتوای سند(های) پیوست‌شده توسط کاربر:\n" + "\n\n".join(doc_blocks)

    if image_attachments:
        user_content = [{"type": "text", "text": turn_text or "این تصویر(ها) را بررسی کن."}]
        for att in image_attachments:
            url = _image_data_url(att)
            if url:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        user_content = turn_text or "(پیوست)"

    return {
        "system_text": system_text,
        "history_msgs": history_msgs,
        "user_content": user_content,
        "has_doc_attachments": bool(doc_blocks),
        "query": user_text or "محتوای سند پیوست",
    }


def _resolve_reply_to(session, request):
    """Return (reply_to_message | None, quote_text). The quote prefers a posted
    snippet (a selected sentence, ChatGPT-style) and falls back to the whole
    message content."""
    rid = (request.POST.get('reply_to') or '').strip()
    if not rid:
        return None, None
    msg = session.messages.filter(pk=rid).first()
    if not msg:
        return None, None
    quote = (request.POST.get('reply_quote') or '').strip() or (msg.content or '').strip()
    return msg, quote


@login_required
@require_POST
def api_send_message(request, session_id):
    """Non-streaming fallback: runs the same RAG pipeline, returns full JSON."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    user_text = (request.POST.get('message') or '').strip()
    uploads = request.FILES.getlist('attachments')
    if not user_text and not uploads:
        return JsonResponse({"ok": False, "error": "empty"}, status=400)

    is_first = not session.messages.filter(role=ChatMessage.Role.USER).exists()
    reply_to, quote = _resolve_reply_to(session, request)
    user_msg, images, docs = _persist_turn(session, user_text, uploads)
    if reply_to:
        user_msg.reply_to = reply_to
        user_msg.reply_quote = (quote or '')[:2000]
        user_msg.save(update_fields=["reply_to", "reply_quote"])
    prep = _prepare_turn(session, user_text, images, docs, quote=quote)

    from django.utils import timezone
    reply, sources = "", []
    try:
        for ev in agent_runtime.run_agentic_answer(
                doctor=session.doctor, session=session, user=request.user,
                query=prep["query"], system_text=prep["system_text"],
                history_msgs=prep["history_msgs"], user_content=prep["user_content"],
                enabled_keys=session.doctor.agents or [],
                now_iso=timezone.localtime().isoformat(timespec="minutes")):
            if ev["type"] == "done":
                reply, sources = ev.get("answer", ""), ev.get("sources", [])
            elif ev["type"] == "error":
                reply = ev["message"]
    except Exception:
        reply = "متاسفم—الان به مدل پزشکی دسترسی ندارم. لطفاً کمی بعد دوباره تلاش کنید."
    reply = reply or "پاسخی دریافت نشد. لطفاً دوباره تلاش کنید."

    ChatMessage.objects.create(session=session, role=ChatMessage.Role.ASSISTANT,
                               content=reply, sources=sources)

    new_title = None
    if is_first:
        new_title = _generate_title(user_text) or session.title
        session.title = new_title
        session.save(update_fields=["title", "updated_at"])
    else:
        session.save(update_fields=["updated_at"])

    return JsonResponse({
        "ok": True, "reply": reply, "title": new_title, "sources": sources,
        "attachments": [{"kind": a.kind, "url": a.file.url, "name": a.original_name}
                        for a in (images + docs)],
    })


@login_required
@require_POST
def api_stream_message(request, session_id):
    """Stream the assistant reply as Server-Sent Events, with live search stages
    and numbered sources for inline [n] citations (ROP-assistant style)."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)

    # Regenerate: re-answer the last user turn with a fresh assistant reply.
    if (request.POST.get('regenerate') or '') == '1':
        return _regenerate_stream(request, session)

    user_text = (request.POST.get('message') or '').strip()
    uploads = request.FILES.getlist('attachments')
    if not user_text and not uploads:
        return JsonResponse({"ok": False, "error": "empty"}, status=400)

    is_first = not session.messages.filter(role=ChatMessage.Role.USER).exists()
    reply_to, quote = _resolve_reply_to(session, request)
    user_msg, images, docs = _persist_turn(session, user_text, uploads)
    if reply_to:
        user_msg.reply_to = reply_to
        user_msg.reply_quote = (quote or '')[:2000]
        user_msg.save(update_fields=["reply_to", "reply_quote"])
    prep = _prepare_turn(session, user_text, images, docs, quote=quote)
    attachments = [{"kind": a.kind, "url": a.file.url, "name": a.original_name}
                   for a in (images + docs)]
    return _stream_response(session, prep, is_first=is_first, title_seed=user_text,
                            attachments=attachments, user_msg_id=user_msg.pk)


def _regenerate_stream(request, session):
    """Delete the target assistant message (and anything after it) and stream a
    fresh answer to the last user turn — no new user message is created."""
    mid = (request.POST.get('message_id') or '').strip()
    target = session.messages.filter(pk=mid, role=ChatMessage.Role.ASSISTANT).first()
    if target:
        session.messages.filter(pk__gte=target.pk).delete()
    last_user = (session.messages.filter(role=ChatMessage.Role.USER)
                 .order_by('-created_at', '-pk').first())
    if not last_user:
        return JsonResponse({"ok": False, "error": "no_turn"}, status=400)
    atts = list(last_user.attachments.all())
    images = [a for a in atts if a.kind == ChatAttachment.Kind.IMAGE][:MAX_IMAGES]
    docs = [a for a in atts if a.kind == ChatAttachment.Kind.DOCUMENT]
    quote = last_user.reply_quote or (last_user.reply_to.content if last_user.reply_to else None)
    prep = _prepare_turn(session, last_user.content, images, docs, quote=quote)
    return _stream_response(session, prep, is_first=False, title_seed="",
                            attachments=[], user_msg_id=None)


def _stream_response(session, prep, is_first, title_seed, attachments, user_msg_id):
    """Build the SSE StreamingHttpResponse shared by send + regenerate."""
    import json
    from django.http import StreamingHttpResponse
    from django.utils import timezone

    def sse(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    now_iso = timezone.localtime().isoformat(timespec="minutes")

    def generate():
        yield sse({"attachments": attachments, "user_msg_id": user_msg_id})
        answer, sources, error_msg = "", [], None
        try:
            for ev in agent_runtime.run_agentic_answer(
                    doctor=session.doctor, session=session, user=session.user,
                    query=prep["query"], system_text=prep["system_text"],
                    history_msgs=prep["history_msgs"], user_content=prep["user_content"],
                    enabled_keys=session.doctor.agents or [], now_iso=now_iso):
                t = ev.get("type")
                if t == "token":
                    # `delta` alias keeps the voice-mode consumer working.
                    yield sse({"type": "token", "text": ev["text"], "delta": ev["text"]})
                elif t == "stage":
                    yield sse(ev)
                elif t == "ui":
                    yield sse(ev)
                elif t == "error":
                    error_msg = ev["message"]
                    yield sse(ev)
                elif t == "done":
                    answer, sources = ev.get("answer", ""), ev.get("sources", [])
        except Exception:
            error_msg = error_msg or "متاسفم—الان به مدل پزشکی دسترسی ندارم. لطفاً کمی بعد دوباره تلاش کنید."
            yield sse({"type": "error", "message": error_msg})

        reply = answer or error_msg or "پاسخی دریافت نشد."
        assistant_msg = ChatMessage.objects.create(
            session=session, role=ChatMessage.Role.ASSISTANT,
            content=reply, sources=sources if answer else [])

        # Title the conversation from the first message — streamed word-by-word.
        new_title = None
        if is_first and title_seed:
            pieces = []
            try:
                for piece in LLMClient().chat_stream(_title_messages(title_seed), temperature=0.3):
                    pieces.append(piece)
                    yield sse({"title_delta": piece})
            except Exception:
                pass
            new_title = _clean_title("".join(pieces)) or _fallback_title(title_seed)
            session.title = new_title
            session.save(update_fields=["title", "updated_at"])
        else:
            session.save(update_fields=["updated_at"])
        yield sse({"type": "done", "done": True, "sources": sources,
                   "title": new_title, "message_id": assistant_msg.pk})

    resp = StreamingHttpResponse(generate(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


# Persian / English neural voices (edge-tts) by gender, offline espeak-ng fallback.
TTS_VOICES = {
    "fa": {"male": "fa-IR-FaridNeural", "female": "fa-IR-DilaraNeural"},
    "en": {"male": "en-US-GuyNeural",   "female": "en-US-AriaNeural"},
}


def _pick_voice(lang, gender):
    code = "fa" if str(lang).startswith("fa") else "en"
    g = "female" if str(gender).lower().startswith("f") else "male"
    return TTS_VOICES[code][g]


def _edge_tts_bytes(text, voice):
    import asyncio
    import edge_tts

    async def run():
        data = b""
        comm = edge_tts.Communicate(text, voice)
        async for ch in comm.stream():
            if ch["type"] == "audio":
                data += ch["data"]
        return data

    return asyncio.run(run())


def _edge_tts_timed(text, voice):
    """Like _edge_tts_bytes, but also collects per-word timings from edge-tts
    WordBoundary events so the 3D talking-head can lip-sync. Offsets/durations
    come in 100-nanosecond ticks -> convert to milliseconds."""
    import asyncio
    import edge_tts

    async def run():
        data = b""
        words, wtimes, wdurations = [], [], []
        try:
            comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
        except TypeError:
            # older edge-tts builds always emit WordBoundary and lack the kwarg
            comm = edge_tts.Communicate(text, voice)
        async for ch in comm.stream():
            if ch["type"] == "audio":
                data += ch["data"]
            elif ch["type"] == "WordBoundary":
                words.append(ch["text"])
                wtimes.append(ch["offset"] / 10000.0)
                wdurations.append(ch["duration"] / 10000.0)
        return data, words, wtimes, wdurations

    return asyncio.run(run())


def _espeak_bytes(text, lang):
    import subprocess
    import tempfile
    voice = "fa" if str(lang).startswith("fa") else "en"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        subprocess.run(["espeak-ng", "-v", voice, "-s", "150", "-w", path, text],
                       check=True, timeout=30)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@login_required
@require_POST
def api_rename_session(request, session_id):
    """Rename a conversation (editable chat-history title)."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    title = (request.POST.get('title') or '').strip()[:160]
    if not title:
        return JsonResponse({"ok": False, "error": "empty"}, status=400)
    session.title = title
    session.save(update_fields=["title"])
    return JsonResponse({"ok": True, "title": title})


@login_required
@require_POST
def api_pin_session(request, session_id):
    """Toggle the pinned flag on a conversation (pinned chats sort to the top)."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.pinned = not session.pinned
    session.save(update_fields=["pinned"])
    return JsonResponse({"ok": True, "pinned": session.pinned})


@login_required
@require_POST
def api_delete_session(request, session_id):
    """Delete a conversation. If the current chat is deleted, tell the client
    where to navigate next (the newest remaining chat, or a fresh one)."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    doctor = session.doctor
    session.delete()
    nxt = (ChatSession.objects
           .filter(user=request.user, doctor=doctor)
           .order_by('-pinned', '-updated_at').first())
    if nxt:
        redirect_url = reverse('doctors_marketplace:chat', args=[nxt.id])
    else:
        redirect_url = reverse('doctors_marketplace:doctor_detail', args=[doctor.slug])
    return JsonResponse({"ok": True, "redirect": redirect_url})


@login_required
@require_POST
def api_edit_message(request, session_id):
    """Edit a user message: truncate this message and everything after it, so the
    client can resend the edited text as a fresh turn (ChatGPT-style fork)."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    mid = (request.POST.get('message_id') or '').strip()
    msg = session.messages.filter(pk=mid, role=ChatMessage.Role.USER).first()
    if not msg:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    # Delete this user message and every message created after it in this chat.
    session.messages.filter(pk__gte=msg.pk).delete()
    return JsonResponse({"ok": True})


@login_required
@require_POST
def api_feedback(request, session_id):
    """Save 👍/👎 feedback on an assistant answer. value: up | down | clear."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    mid = (request.POST.get('message_id') or '').strip()
    msg = session.messages.filter(pk=mid, role=ChatMessage.Role.ASSISTANT).first()
    if not msg:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    msg.feedback = {"up": 1, "down": -1}.get(request.POST.get('value'), 0)
    msg.save(update_fields=["feedback"])
    return JsonResponse({"ok": True, "feedback": msg.feedback})


def _search_snippet(text, q, radius=45):
    """A short excerpt of `text` around the first case-insensitive hit of `q`."""
    if not text:
        return ""
    low = text.lower()
    i = low.find(q.lower())
    if i < 0:
        return text[:2 * radius].strip()
    start = max(0, i - radius)
    end = min(len(text), i + len(q) + radius)
    out = text[start:end].strip().replace("\n", " ")
    return ("…" if start else "") + out + ("…" if end < len(text) else "")


@login_required
def api_search_sessions(request, session_id):
    """Search this user's conversations with the current doctor by title or any
    message content. Returns matches for the sidebar search box."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    q = (request.GET.get('q') or '').strip()
    if not q:
        return JsonResponse({"ok": True, "results": []})

    base = ChatSession.objects.filter(user=request.user, doctor=session.doctor)
    matches = (base.filter(Q(title__icontains=q) | Q(messages__content__icontains=q))
               .distinct().order_by('-pinned', '-updated_at')[:30])

    results = []
    for s in matches:
        snippet = ""
        hit = (s.messages.exclude(role=ChatMessage.Role.SYSTEM)
               .filter(content__icontains=q).first())
        if hit:
            snippet = _search_snippet(hit.content, q)
        results.append({
            "id": str(s.id),
            "title": s.title or "New chat",
            "url": reverse('doctors_marketplace:chat', args=[s.id]),
            "snippet": snippet,
            "pinned": s.pinned,
            "active": s.id == session.id,
        })
    return JsonResponse({"ok": True, "results": results})


@login_required
def api_tts(request):
    """
    Server-side text-to-speech so Persian works regardless of the user's browser
    voices. Tries edge-tts (neural) and falls back to offline espeak-ng.
    """
    from django.http import HttpResponse, HttpResponseBadRequest

    text = (request.GET.get('text') or request.POST.get('text') or '').strip()
    lang = (request.GET.get('lang') or request.POST.get('lang') or 'fa').strip().lower()
    if not text:
        return HttpResponseBadRequest("empty")
    text = text[:600]  # one sentence/chunk per call
    gender = (request.GET.get('gender') or request.POST.get('gender') or 'male')
    voice = _pick_voice(lang, gender)

    try:
        audio = _edge_tts_bytes(text, voice)
        if audio:
            return HttpResponse(audio, content_type="audio/mpeg")
    except Exception:
        pass
    try:
        audio = _espeak_bytes(text, lang)
        return HttpResponse(audio, content_type="audio/wav")
    except Exception:
        return HttpResponse(status=503)


@login_required
def api_tts_timed(request):
    """TTS that also returns per-word timings so the 3D talking-head (TalkingHead)
    can lip-sync to our own edge-tts audio. Returns JSON:
        {audio: <base64 mp3>, mime, words: [], wtimes: [ms], wdurations: [ms]}
    English gets accurate visemes (TalkingHead's `en` lipsync module aligns to the
    word times); Persian still plays correct audio with approximate mouth movement
    until a Persian viseme module is added.
    """
    import base64

    text = (request.GET.get('text') or request.POST.get('text') or '').strip()
    lang = (request.GET.get('lang') or 'fa').strip().lower()
    if not text:
        return JsonResponse({"error": "empty"}, status=400)
    text = text[:600]
    gender = (request.GET.get('gender') or request.POST.get('gender') or 'male')
    voice = _pick_voice(lang, gender)

    try:
        audio, words, wtimes, wdurations = _edge_tts_timed(text, voice)
        if audio:
            return JsonResponse({
                "audio": base64.b64encode(audio).decode('ascii'),
                "mime": "audio/mpeg",
                "words": words,
                "wtimes": wtimes,
                "wdurations": wdurations,
            })
    except Exception:
        pass
    return JsonResponse({"error": "tts_failed"}, status=503)


# -------- STUDIO (superusers) --------
@login_required
@user_passes_test(_is_superuser)
def studio_index(request):
    doctors = Doctor.objects.all().order_by('-created_at')
    return render(request, 'doctors_marketplace/studio/index2.html', {"doctors": doctors})


def _posted_agent_keys(request):
    """Validated list of enabled agent keys from the studio form checkboxes."""
    from . import agents as agents_pkg
    return agents_pkg.valid_keys(request.POST.getlist('agents'))


def _studio_context(form, doctor):
    from . import agents as agents_pkg
    return {
        "form": form, "doctor": doctor,
        "agent_catalog": agents_pkg.catalog(),
        "enabled_agents": (doctor.agents if doctor else []) or [],
    }


@login_required
@user_passes_test(_is_superuser)
def studio_new(request):
    if request.method == 'POST':
        form = DoctorForm(request.POST, request.FILES)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.agents = _posted_agent_keys(request)
            doc.save()
            messages.success(request, "Doctor created successfully.")
            return redirect('doctors_marketplace:studio_edit', slug=doc.slug)
    else:
        form = DoctorForm()
    return render(request, 'doctors_marketplace/studio/new2.html', _studio_context(form, None))


@login_required
@user_passes_test(_is_superuser)
def studio_edit(request, slug):
    doctor = get_object_or_404(Doctor, slug=slug)
    if request.method == 'POST':
        form = DoctorForm(request.POST, request.FILES, instance=doctor)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.agents = _posted_agent_keys(request)
            doc.save()
            messages.success(request, "Changes saved.")
            return redirect('doctors_marketplace:studio_edit', slug=doctor.slug)
    else:
        form = DoctorForm(instance=doctor)
    return render(request, 'doctors_marketplace/studio/new2.html', _studio_context(form, doctor))


@login_required
@user_passes_test(_is_superuser)
@require_POST
def studio_copilot(request):
    """Conversational helper that fills the assistant form and picks agents."""
    import json
    from .services import studio_copilot as copilot
    try:
        data = json.loads(request.body or "{}")
    except (ValueError, TypeError):
        return JsonResponse({"error": "bad_request"}, status=400)
    history = data.get("messages") or []
    form_state = data.get("form") or {}
    if not isinstance(history, list) or not isinstance(form_state, dict):
        return JsonResponse({"error": "bad_request"}, status=400)
    try:
        result = copilot.run_copilot(history, form_state)
    except Exception as e:  # noqa: BLE001 - surface a clean error to the panel
        return JsonResponse({"error": "copilot_failed", "detail": str(e)}, status=503)
    return JsonResponse(result)


@login_required
@user_passes_test(_is_superuser)
@require_POST
def studio_copilot_avatar(request):
    """Generate a realistic doctor face for the drafted assistant."""
    import json
    from .services import image_gen
    try:
        data = json.loads(request.body or "{}")
    except (ValueError, TypeError):
        return JsonResponse({"error": "bad_request"}, status=400)
    url = image_gen.generate_doctor_face(
        name=(data.get("name") or "").strip(),
        specialization=(data.get("specialization") or "").strip(),
        persona=(data.get("persona") or "").strip(),
    )
    if not url:
        return JsonResponse({"error": "image_failed"}, status=503)
    return JsonResponse({"url": url})


@login_required
@user_passes_test(_is_superuser)
@require_POST
def studio_delete(request, slug):
    doctor = get_object_or_404(Doctor, slug=slug)
    doctor.delete()
    messages.success(request, "Doctor deleted.")
    return redirect('doctors_marketplace:studio_index')


@login_required
@user_passes_test(_is_superuser)
def studio_kb(request, slug):
    doctor = get_object_or_404(Doctor, slug=slug)
    items = doctor.knowledge_items.order_by('-created_at')
    if request.method == 'POST':
        form = KnowledgeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.doctor = doctor
            obj.save()
            messages.success(request, "Knowledge file uploaded. Indexing runs in the background.")
            return redirect('doctors_marketplace:studio_kb', slug=slug)
    else:
        form = KnowledgeUploadForm()
    return render(request, 'doctors_marketplace/studio/kb2.html', {"doctor": doctor, "items": items, "form": form})


@login_required
@user_passes_test(_is_superuser)
@require_POST
def studio_kb_delete(request, slug, pk):
    doctor = get_object_or_404(Doctor, slug=slug)
    item = get_object_or_404(DoctorKnowledge, pk=pk, doctor=doctor)
    item.delete()
    messages.success(request, "Knowledge file deleted.")
    return redirect('doctors_marketplace:studio_kb', slug=slug)


@login_required
@user_passes_test(_is_superuser)
@require_POST
def studio_kb_reindex(request, slug):
    """Rebuild the doctor's whole vector index from all knowledge files."""
    from .services.rag import rebuild_doctor_index
    doctor = get_object_or_404(Doctor, slug=slug)
    try:
        ok, msg = rebuild_doctor_index(doctor)
    except Exception as e:  # noqa: BLE001
        ok, msg = False, str(e)
    messages.success(request, f"Index rebuilt: {msg}") if ok else \
        messages.warning(request, f"Rebuild failed: {msg}")
    return redirect('doctors_marketplace:studio_kb', slug=slug)
