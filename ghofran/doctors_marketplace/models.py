"""SQLAlchemy ORM models for the Doctors-Marketplace service.

Mirrors the Django models: ``Doctor``, ``ChatSession``, ``ChatMessage`` and
``DoctorKnowledge``. Differences from Django:

* ``ChatSession.id`` is a UUID *string* (portable across SQLite/Postgres).
* ``user_id`` is a plain string identifier rather than a FK into Django's auth
  table, keeping this service independent of USAC. The caller supplies whatever
  stable user id it uses (e.g. the ``sub`` from a USAC JWT).
* ``vector_dir`` / knowledge paths are computed from config, not Django settings.
"""
from __future__ import annotations

import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .config import get_settings
from .database import Base

# Chat message roles (same string values as the original TextChoices).
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def slugify_filename(name: str) -> str:
    """Normalise a filename, preserving non-ASCII (Persian) letters.

    Ported from the original ``slugify_filename`` so uploaded KB files land with
    the same human-readable names.
    """
    name = unicodedata.normalize("NFKC", name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "-", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


class Doctor(Base):
    """A specialised chat assistant with a persona and optional knowledge base."""

    __tablename__ = "dm_doctor"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # English (primary) fields.
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(140), unique=True, index=True, nullable=False)
    specialization: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    persona: Mapped[str] = mapped_column(String(20), default="kind", nullable=False)
    headline: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    bio: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Persian fields.
    name_fa: Mapped[str] = mapped_column(String(140), default="", nullable=False)
    specialization_fa: Mapped[str] = mapped_column(String(140), default="", nullable=False)
    headline_fa: Mapped[str] = mapped_column(String(220), default="", nullable=False)
    bio_fa: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tags_fa: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Long system prompt used to seed each chat.
    system_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    knowledge_items: Mapped[list["DoctorKnowledge"]] = relationship(
        back_populates="doctor", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="doctor", cascade="all, delete-orphan"
    )

    def vector_dir(self) -> str:
        """Return (creating if needed) this doctor's FAISS index directory."""
        path = os.path.join(get_settings().vector_root, self.slug)
        os.makedirs(path, exist_ok=True)
        return path

    def knowledge_dir(self) -> str:
        """Return (creating if needed) this doctor's uploaded-files directory."""
        path = os.path.join(get_settings().knowledge_root, self.slug)
        os.makedirs(path, exist_ok=True)
        return path


class ChatSession(Base):
    """A conversation between a user and a doctor."""

    __tablename__ = "dm_chat_session"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("dm_doctor.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    doctor: Mapped[Doctor] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    """A single turn (system/user/assistant) inside a chat session."""

    __tablename__ = "dm_chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("dm_chat_session.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class DoctorKnowledge(Base):
    """An uploaded knowledge file feeding a doctor's RAG index."""

    __tablename__ = "dm_doctor_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("dm_doctor.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    # Absolute path to the stored file on disk.
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    doctor: Mapped[Doctor] = relationship(back_populates="knowledge_items")
