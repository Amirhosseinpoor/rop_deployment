"""SQLAlchemy ORM models for the USAC service.

These mirror the Django models (``Company``, ``UserProfile``, ``Invitation``)
plus a ``User`` table (Django provided ``auth.User`` for free; a standalone
service must own its users).

Roles are kept as plain strings (matching Django's choices) rather than an Enum
column so the data is portable across databases without custom types.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

# Role constants — identical values to the Django ``UserProfile.ROLE_*``.
ROLE_MANAGER = "manager"
ROLE_DOCTOR = "doctor"
ROLE_EMPLOYEE = "employee"
ROLE_CHOICES = (ROLE_MANAGER, ROLE_DOCTOR, ROLE_EMPLOYEE)


def _utcnow() -> datetime:
    """Timezone-aware UTC now (avoids the deprecated naive ``utcnow``)."""
    return datetime.now(timezone.utc)


class User(Base):
    """A login identity. Replaces Django's built-in ``auth.User``."""

    __tablename__ = "usac_user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(254), default="", nullable=False)
    # Bcrypt/argon hash — never the plaintext password.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    date_joined: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    profile: Mapped["UserProfile"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Company(Base):
    """An organisation, owned (one-to-one) by a manager user."""

    __tablename__ = "usac_company"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    address: Mapped[str] = mapped_column(Text, default="", nullable=False)
    email: Mapped[str] = mapped_column(String(254), default="", nullable=False)
    phone: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # PROTECT in Django -> we simply keep a non-nullable FK and never cascade-delete.
    manager_id: Mapped[int] = mapped_column(
        ForeignKey("usac_user.id"), unique=True, nullable=False
    )

    manager: Mapped[User] = relationship("User")
    members: Mapped[list["UserProfile"]] = relationship(back_populates="company")


class UserProfile(Base):
    """Extends a user with role, company membership and contact details."""

    __tablename__ = "usac_userprofile"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("usac_user.id"), unique=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    national_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("usac_company.id"), nullable=True
    )
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="profile")
    company: Mapped[Company | None] = relationship(back_populates="members")


class Invitation(Base):
    """A manager-issued, single-use permission for staff to self-register.

    A staff signup only succeeds if a matching (national_code, role) invitation
    exists for some company and has not yet been used.
    """

    __tablename__ = "usac_invitation"
    __table_args__ = (
        # Mirrors Django's ``unique_together = ('company', 'national_code')``.
        UniqueConstraint("company_id", "national_code", name="uq_company_national_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("usac_company.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    national_code: Mapped[str] = mapped_column(String(10), nullable=False)
    note: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    used_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("usac_user.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    company: Mapped[Company] = relationship("Company")

    def mark_used(self, user: User) -> None:
        """Bind this invitation to ``user`` and stamp the usage time."""
        self.used_by_id = user.id
        self.used_at = _utcnow()
