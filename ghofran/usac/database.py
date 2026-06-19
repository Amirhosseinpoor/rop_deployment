"""SQLAlchemy engine / session plumbing for the USAC service.

Isolated in its own module so both the models and the FastAPI dependency layer
import the *same* ``Base`` and ``SessionLocal`` without circular imports.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

_settings = get_settings()

# SQLite needs ``check_same_thread=False`` to be usable from FastAPI's threadpool;
# the flag is harmless/ignored for other backends.
_connect_args = (
    {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(_settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model in this service."""


def init_db() -> None:
    """Create all tables. Idempotent; safe to call at every start-up."""
    # Import models for their side effect of registering with ``Base.metadata``.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a request-scoped session and closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
