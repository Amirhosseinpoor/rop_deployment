# doctors_marketplace/views.py
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from django.contrib import messages

from .models import Doctor, ChatSession, ChatMessage, DoctorKnowledge
from .forms import DoctorForm, KnowledgeUploadForm
from .services.llm import LLMClient
from .services.rag import retrieve_context
def _is_superuser(user):
    return user.is_authenticated and user.is_superuser

# -------- MARKET (public) --------
def market_index(request):
    doctors = Doctor.objects.filter(is_active=True).order_by('name_fa', 'name')
    return render(request, 'doctors_marketplace/market_index.html', {"doctors": doctors})

def doctor_detail(request, slug):
    doctor = get_object_or_404(Doctor, slug=slug, is_active=True)
    if request.method == 'POST':
        if not request.user.is_authenticated:
            return redirect('account_login')
        title = f"گفتگو با {doctor.name_fa or ('دکتر ' + doctor.name)}"
        session = ChatSession.objects.create(user=request.user, doctor=doctor, title=title)
        # Seed system from DB
        sys_text = doctor.system_prompt or (
            "شما یک دستیار پزشکی فارسی‌زبان هستید. به احوالپرسی پاسخ کوتاه بدهید و سپس به موضوع تخصص پزشک برگردید."
        )
        ChatMessage.objects.create(session=session, role=ChatMessage.Role.SYSTEM, content=sys_text)
        return redirect('doctors_marketplace:chat', session_id=session.id)
    return render(request, 'doctors_marketplace/doctor_detail.html', {"doctor": doctor})

# -------- CHAT --------
@login_required
def chat_view(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages_qs = session.messages.all()
    history = ChatSession.objects.filter(user=request.user, doctor=session.doctor).order_by('-updated_at')[:30]
    return render(request, 'doctors_marketplace/chat.html', {
        "session": session,
        "messages": messages_qs,
        "history": history,
    })

@login_required
@require_POST
def api_send_message(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    user_text = (request.POST.get('message') or '').strip()
    if not user_text:
        return JsonResponse({"ok": False, "error": "empty"}, status=400)

    ChatMessage.objects.create(session=session, role=ChatMessage.Role.USER, content=user_text)

    system = session.messages.filter(role=ChatMessage.Role.SYSTEM).first()
    system_text = system.content if system else (session.doctor.system_prompt or "شما یک دستیار پزشکی فارسی‌زبان هستید.")
    history = session.messages.exclude(role=ChatMessage.Role.SYSTEM).order_by('created_at')
    msgs = [{"role": "system", "content": system_text}] + [{"role": m.role, "content": m.content} for m in history]
    rag_docs = retrieve_context(session.doctor, user_text, k=4)
    context_block = ""
    if rag_docs:
        bullets = []
        for i, d in enumerate(rag_docs, 1):
            src = (d.metadata or {}).get("title") or "KB"
            snippet = (d.page_content or "").strip()
            snippet = snippet[:900]  # keep prompt small
            bullets.append(f"[{i}] ({src})\n{snippet}")
        context_block = "منابع داخلی پزشک:\n" + "\n\n".join(bullets)

    # Insert contextual system message *after* the main system
    if context_block:
        msgs.insert(1, {"role": "system",
                        "content": f"از اطلاعات زیر برای پاسخ دقیق‌تر استفاده کن؛ اگر مرتبط نبود، نادیده بگیر:\n{context_block}"})
    try:
        client = LLMClient()
        reply = client.chat(msgs)
    except Exception:
        reply = "متاسفم—الان به مدل پزشکی دسترسی ندارم. لطفاً دوباره تلاش کنید."

    ChatMessage.objects.create(session=session, role=ChatMessage.Role.ASSISTANT, content=reply)
    return JsonResponse({"ok": True, "reply": reply})

# -------- STUDIO (superusers) --------
@login_required
@user_passes_test(_is_superuser)
def studio_index(request):
    doctors = Doctor.objects.all().order_by('-created_at')
    return render(request, 'doctors_marketplace/studio/index.html', {"doctors": doctors})

@login_required
@user_passes_test(_is_superuser)
def studio_new(request):
    if request.method == 'POST':
        form = DoctorForm(request.POST, request.FILES)
        if form.is_valid():
            doc = form.save()
            messages.success(request, "پزشک با موفقیت ایجاد شد.")
            return redirect('doctors_marketplace:studio_edit', slug=doc.slug)
    else:
        form = DoctorForm()
    return render(request, 'doctors_marketplace/studio/new.html', {"form": form, "doctor": None})

@login_required
@user_passes_test(_is_superuser)
def studio_edit(request, slug):
    doctor = get_object_or_404(Doctor, slug=slug)
    if request.method == 'POST':
        form = DoctorForm(request.POST, request.FILES, instance=doctor)
        if form.is_valid():
            form.save()
            messages.success(request, "تغییرات ذخیره شد.")
            return redirect('doctors_marketplace:studio_edit', slug=doctor.slug)
    else:
        form = DoctorForm(instance=doctor)
    return render(request, 'doctors_marketplace/studio/new.html', {"form": form, "doctor": doctor})

@login_required
@user_passes_test(_is_superuser)
@require_POST
def studio_delete(request, slug):
    doctor = get_object_or_404(Doctor, slug=slug)
    doctor.delete()
    messages.success(request, "پزشک حذف شد.")
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
            messages.success(request, "فایل دانش بارگذاری شد.")
            return redirect('doctors_marketplace:studio_kb', slug=slug)
    else:
        form = KnowledgeUploadForm()
    return render(request, 'doctors_marketplace/studio/kb.html', {"doctor": doctor, "items": items, "form": form})

@login_required
@user_passes_test(_is_superuser)
@require_POST
def studio_kb_delete(request, slug, pk):
    doctor = get_object_or_404(Doctor, slug=slug)
    item = get_object_or_404(DoctorKnowledge, pk=pk, doctor=doctor)
    item.delete()
    messages.success(request, "فایل دانش حذف شد.")
    return redirect('doctors_marketplace:studio_kb', slug=slug)
