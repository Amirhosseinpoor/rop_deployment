"""Business logic for the Doctors-Marketplace service (framework-free).

Operates on an explicit SQLAlchemy ``Session`` and raises :class:`ServiceError`
on domain problems. Consolidates the logic from the Django views, signals and
tasks: seeding doctors, opening sessions, the RAG+LLM chat turn, doctor CRUD, and
synchronous knowledge-base indexing (the original did this via a Celery task; a
single service does it inline / in a background task instead).
"""
from __future__ import annotations

import os
import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models
from .llm import LLMClient
from .models import ChatMessage, ChatSession, Doctor, DoctorKnowledge
from .prompts import DOCTOR_DEFS
from .rag import index_file, retrieve_context

# Default system prompt used when a doctor has none (matches the Django default).
_DEFAULT_SYSTEM = (
    "شما یک دستیار پزشکی فارسی‌زبان هستید. به احوالپرسی پاسخ کوتاه بدهید و سپس "
    "به موضوع تخصص پزشک برگردید."
)


class ServiceError(Exception):
    """Domain-level error with a ``code`` ('not_found' | 'bad_request')."""

    def __init__(self, message: str, code: str = "bad_request"):
        super().__init__(message)
        self.message = message
        self.code = code


def _slugify(value: str) -> str:
    """ASCII slug from a name (lower, hyphen-separated). Mirrors Django slugify."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


# --------------------------------------------------------------------------- #
# Doctors (public read + seeding)
# --------------------------------------------------------------------------- #
def list_active_doctors(db: Session) -> list[Doctor]:
    """Return all active doctors, ordered by Persian then English name."""
    return list(
        db.scalars(
            select(Doctor).where(Doctor.is_active.is_(True)).order_by(Doctor.name_fa, Doctor.name)
        )
    )


def get_doctor_by_slug(db: Session, slug: str, *, active_only: bool = True) -> Doctor:
    """Fetch a doctor by slug or raise a not-found error."""
    stmt = select(Doctor).where(Doctor.slug == slug)
    if active_only:
        stmt = stmt.where(Doctor.is_active.is_(True))
    doctor = db.scalar(stmt)
    if doctor is None:
        raise ServiceError(f"Doctor '{slug}' not found.", code="not_found")
    return doctor


def seed_doctors(db: Session) -> int:
    """Create any missing doctors from the static ``DOCTOR_DEFS`` catalogue.

    Idempotent: existing slugs are skipped. Returns the number created. This
    replaces the Django ``seed_doctors`` management command.
    """
    created = 0
    for d in DOCTOR_DEFS:
        if db.scalar(select(Doctor).where(Doctor.slug == d["slug"])) is not None:
            continue
        db.add(
            Doctor(
                name=d.get("name", ""),
                slug=d["slug"],
                specialization=d.get("specialization", ""),
                persona=d.get("persona", "kind"),
                headline=d.get("headline", ""),
                bio=d.get("bio", ""),
                name_fa=d.get("name_fa", ""),
                specialization_fa=d.get("specialization_fa", ""),
                headline_fa=d.get("headline_fa", ""),
                bio_fa=d.get("bio_fa", ""),
                tags_fa=d.get("tags_fa", ""),
                # The catalogue uses the key "system" for the long prompt.
                system_prompt=d.get("system", ""),
            )
        )
        created += 1
    if created:
        db.commit()
    return created


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def create_session(db: Session, *, user_id: str, slug: str) -> ChatSession:
    """Open a new chat session with a doctor and seed its system message."""
    doctor = get_doctor_by_slug(db, slug)
    title = f"گفتگو با {doctor.name_fa or ('دکتر ' + doctor.name)}"
    session = ChatSession(user_id=user_id, doctor_id=doctor.id, title=title)
    db.add(session)
    db.flush()
    # Seed the conversation with the doctor's system prompt.
    db.add(
        ChatMessage(
            session_id=session.id,
            role=models.ROLE_SYSTEM,
            content=doctor.system_prompt or _DEFAULT_SYSTEM,
        )
    )
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, *, session_id: str, user_id: str) -> ChatSession:
    """Fetch a session owned by ``user_id`` or raise not-found."""
    session = db.scalar(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
    )
    if session is None:
        raise ServiceError("Chat session not found.", code="not_found")
    return session


def send_message(db: Session, *, session_id: str, user_id: str, user_text: str) -> str:
    """Record a user message, query the LLM (RAG-grounded), store & return reply.

    This is the heart of the service, ported from ``api_send_message``:
    1. persist the user's message;
    2. assemble the message list (system prompt + history);
    3. retrieve up to 4 relevant KB chunks and inject them as an extra system
       message so the model can ground its answer;
    4. call the LLM (failing soft to a friendly Persian message);
    5. persist and return the assistant reply.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        raise ServiceError("Message is empty.")

    session = get_session(db, session_id=session_id, user_id=user_id)
    db.add(ChatMessage(session_id=session.id, role=models.ROLE_USER, content=user_text))
    db.flush()

    # Build the base message list: the system prompt, then the non-system history.
    system_msg = next((m for m in session.messages if m.role == models.ROLE_SYSTEM), None)
    system_text = (
        system_msg.content
        if system_msg
        else (session.doctor.system_prompt or "شما یک دستیار پزشکی فارسی‌زبان هستید.")
    )
    history = [m for m in session.messages if m.role != models.ROLE_SYSTEM]
    msgs: list[dict[str, str]] = [{"role": "system", "content": system_text}]
    msgs += [{"role": m.role, "content": m.content} for m in history]

    # Retrieve doctor-specific knowledge to ground the answer.
    context_block = _build_context_block(session.doctor, user_text)
    if context_block:
        msgs.insert(
            1,
            {
                "role": "system",
                "content": (
                    "از اطلاعات زیر برای پاسخ دقیق‌تر استفاده کن؛ اگر مرتبط نبود، "
                    f"نادیده بگیر:\n{context_block}"
                ),
            },
        )

    # Call the model; degrade gracefully exactly like the original view.
    try:
        reply = LLMClient().chat(msgs)
    except Exception:  # noqa: BLE001 - any LLM/transport failure -> friendly fallback
        reply = "متاسفم—الان به مدل پزشکی دسترسی ندارم. لطفاً دوباره تلاش کنید."

    db.add(ChatMessage(session_id=session.id, role=models.ROLE_ASSISTANT, content=reply))
    db.commit()
    return reply


def _build_context_block(doctor: Doctor, query: str) -> str:
    """Retrieve KB chunks for ``query`` and format them as a prompt block."""
    rag_docs = retrieve_context(doctor.vector_dir(), query, k=4)
    if not rag_docs:
        return ""
    bullets = []
    for i, d in enumerate(rag_docs, 1):
        src = (d.metadata or {}).get("title") or "KB"
        snippet = (d.page_content or "").strip()[:900]  # keep the prompt small
        bullets.append(f"[{i}] ({src})\n{snippet}")
    return "منابع داخلی پزشک:\n" + "\n\n".join(bullets)


# --------------------------------------------------------------------------- #
# Studio (admin CRUD + knowledge base)
# --------------------------------------------------------------------------- #
def list_all_doctors(db: Session) -> list[Doctor]:
    """Return every doctor (active or not), newest first — for the studio view."""
    return list(db.scalars(select(Doctor).order_by(Doctor.created_at.desc())))


def create_doctor(db: Session, data: dict) -> Doctor:
    """Create a doctor from a field dict, auto-generating a slug when absent."""
    slug = (data.get("slug") or "").strip() or _slugify(data.get("name") or data.get("name_fa") or "")
    if not slug:
        raise ServiceError("A name or slug is required to create a doctor.")
    if db.scalar(select(Doctor).where(Doctor.slug == slug)) is not None:
        raise ServiceError(f"Slug '{slug}' already exists.", code="bad_request")
    doctor = Doctor(slug=slug, **{k: v for k, v in data.items() if k != "slug" and v is not None})
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return doctor


def update_doctor(db: Session, slug: str, data: dict) -> Doctor:
    """Update mutable fields of an existing doctor."""
    doctor = get_doctor_by_slug(db, slug, active_only=False)
    for key, value in data.items():
        if value is not None and hasattr(doctor, key) and key != "id":
            setattr(doctor, key, value)
    db.commit()
    db.refresh(doctor)
    return doctor


def delete_doctor(db: Session, slug: str) -> None:
    """Delete a doctor (and, by cascade, their sessions and knowledge rows)."""
    doctor = get_doctor_by_slug(db, slug, active_only=False)
    db.delete(doctor)
    db.commit()


def add_knowledge(db: Session, *, slug: str, title: str, filename: str, content: bytes) -> DoctorKnowledge:
    """Save an uploaded knowledge file, record it, and index it into FAISS.

    Indexing is synchronous here (the Django app deferred it to Celery). For large
    files prefer calling this from a FastAPI ``BackgroundTask`` — the route does.
    """
    doctor = get_doctor_by_slug(db, slug, active_only=False)

    # Build a stable, human-readable filename, preserving the extension.
    base, ext = os.path.splitext(filename)
    safe = f"{models.slugify_filename(title or 'doc')}-{models.slugify_filename(base)}{ext}".strip("-")
    dest = os.path.join(doctor.knowledge_dir(), safe)
    with open(dest, "wb") as f:
        f.write(content)

    item = DoctorKnowledge(doctor_id=doctor.id, title=title, file_path=dest)
    db.add(item)
    db.commit()
    db.refresh(item)

    # Best-effort indexing; failure to index should not lose the uploaded file.
    try:
        index_file(doctor.slug, doctor.vector_dir(), dest, title)
    except Exception as e:  # noqa: BLE001
        # Re-raise as a domain error so the caller can surface a clear message.
        raise ServiceError(f"File saved but indexing failed: {e}")
    return item


def list_knowledge(db: Session, slug: str) -> list[DoctorKnowledge]:
    """Return a doctor's knowledge items, newest first."""
    doctor = get_doctor_by_slug(db, slug, active_only=False)
    return list(
        db.scalars(
            select(DoctorKnowledge)
            .where(DoctorKnowledge.doctor_id == doctor.id)
            .order_by(DoctorKnowledge.created_at.desc())
        )
    )


def delete_knowledge(db: Session, *, slug: str, item_id: int) -> None:
    """Delete a knowledge row (and its file on disk) for a doctor."""
    doctor = get_doctor_by_slug(db, slug, active_only=False)
    item = db.scalar(
        select(DoctorKnowledge).where(
            DoctorKnowledge.id == item_id, DoctorKnowledge.doctor_id == doctor.id
        )
    )
    if item is None:
        raise ServiceError("Knowledge item not found.", code="not_found")
    # Remove the file from disk if present (the FAISS index is left intact; a full
    # rebuild would be needed to evict its chunks — out of scope for a delete).
    try:
        if item.file_path and os.path.exists(item.file_path):
            os.remove(item.file_path)
    except OSError:
        pass
    db.delete(item)
    db.commit()
