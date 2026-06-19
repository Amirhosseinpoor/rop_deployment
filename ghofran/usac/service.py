"""Business logic for the USAC service — pure functions over a DB session.

No FastAPI imports here. Each function takes an explicit SQLAlchemy ``Session``
and raises :class:`ServiceError` on domain violations; the route layer maps those
to HTTP status codes. This mirrors (and consolidates) the logic that lived in the
Django views and forms.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models
from .models import Company, Invitation, User, UserProfile
from .security import hash_password, verify_password


class ServiceError(Exception):
    """Domain-level error with a machine-readable ``code``.

    ``code`` lets the route layer choose an HTTP status without string-matching
    messages: ``conflict`` -> 409, ``not_found`` -> 404, ``auth`` -> 401,
    everything else -> 400.
    """

    def __init__(self, message: str, code: str = "bad_request"):
        super().__init__(message)
        self.message = message
        self.code = code


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #
def get_user_by_username(db: Session, username: str) -> User | None:
    """Return the user with ``username`` or ``None``."""
    return db.scalar(select(User).where(User.username == username))


def get_user_company(db: Session, user: User) -> Company | None:
    """Resolve the user's company via profile, falling back to managed company.

    Faithful to the original ``_get_user_company``: prefer the profile's company,
    else the company this user manages; if found via the fallback, attach it to
    the profile so subsequent lookups are direct.
    """
    profile = user.profile
    if profile and profile.company:
        return profile.company

    managed = db.scalar(select(Company).where(Company.manager_id == user.id))
    if managed and profile:
        profile.company_id = managed.id
        db.flush()
    return managed


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def authenticate(db: Session, username: str, password: str) -> User:
    """Verify credentials and return the user, or raise an auth error."""
    user = get_user_by_username(db, username)
    if user is None or not verify_password(password, user.password_hash):
        raise ServiceError("Invalid credentials", code="auth")
    if not user.is_active:
        raise ServiceError("Account is disabled", code="auth")
    return user


def role_of(user: User) -> str | None:
    """Return the user's role string, or ``None`` if no profile exists yet."""
    return user.profile.role if user.profile else None


# --------------------------------------------------------------------------- #
# Registration flows
# --------------------------------------------------------------------------- #
def _create_user(db: Session, username: str, email: str, password: str) -> User:
    """Create a user, enforcing username uniqueness (case-sensitive like Django)."""
    if get_user_by_username(db, username) is not None:
        raise ServiceError("Username already exists.", code="conflict")
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.flush()  # assign user.id without committing yet
    return user


def signup_manager(
    db: Session,
    *,
    username: str,
    email: str,
    password: str,
    phone: str | None,
    company_name: str,
    company_address: str,
    company_email: str,
    company_phone: str,
) -> User:
    """Register a manager and create the company they will own.

    Mirrors ``ManagerSignupForm.save``: validate unique (case-insensitive)
    company name, create user + manager profile + company, then link them.
    """
    name = company_name.strip()
    existing = db.scalar(
        select(Company).where(func.lower(Company.name) == name.lower())
    )
    if existing is not None:
        raise ServiceError("Company name is already registered.", code="conflict")

    user = _create_user(db, username, email, password)

    company = Company(
        name=name,
        address=company_address,
        email=company_email,
        phone=company_phone,
        manager_id=user.id,
    )
    db.add(company)
    db.flush()

    profile = UserProfile(
        user_id=user.id,
        role=models.ROLE_MANAGER,
        phone=phone or "",
        company_id=company.id,
    )
    db.add(profile)
    db.commit()
    db.refresh(user)
    return user


def signup_staff(
    db: Session,
    *,
    role: str,
    username: str,
    email: str,
    password: str,
    national_code: str,
    phone: str | None,
) -> User:
    """Register a doctor or employee against an active invitation.

    Mirrors ``StaffSignupForm``: the national code must be 10 digits and a
    matching, unused invitation for the requested ``role`` must exist. The
    invitation is consumed on success.
    """
    if role not in (models.ROLE_DOCTOR, models.ROLE_EMPLOYEE):
        raise ServiceError("Staff role must be 'doctor' or 'employee'.")
    if len(national_code) != 10 or not national_code.isdigit():
        raise ServiceError("National code must be exactly 10 digits.")

    invitation = db.scalar(
        select(Invitation).where(
            Invitation.national_code == national_code,
            Invitation.role == role,
            Invitation.used_by_id.is_(None),
        )
    )
    if invitation is None:
        raise ServiceError(
            "No active invitation found for this national code. "
            "Ask your company manager to invite you.",
            code="not_found",
        )

    user = _create_user(db, username, email, password)
    profile = UserProfile(
        user_id=user.id,
        role=role,
        national_code=national_code,
        phone=phone or "",
        company_id=invitation.company_id,
    )
    db.add(profile)
    invitation.mark_used(user)
    db.commit()
    db.refresh(user)
    return user


# --------------------------------------------------------------------------- #
# Manager operations
# --------------------------------------------------------------------------- #
def create_or_update_invitation(
    db: Session, *, manager: User, national_code: str, role: str
) -> Invitation:
    """Issue (or re-target) an invitation scoped to the manager's company.

    Mirrors the invite POST in ``manager_dashboard``: validates inputs, then
    upserts on (company, national_code).
    """
    if role not in models.ROLE_CHOICES:
        raise ServiceError("Invalid role.")
    if len(national_code) != 10 or not national_code.isdigit():
        raise ServiceError("National code must be exactly 10 digits.")

    company = get_user_company(db, manager)
    if company is None:
        raise ServiceError("Your manager account is not linked to a company.", code="not_found")

    invitation = db.scalar(
        select(Invitation).where(
            Invitation.company_id == company.id,
            Invitation.national_code == national_code,
        )
    )
    if invitation is None:
        invitation = Invitation(
            company_id=company.id, national_code=national_code, role=role
        )
        db.add(invitation)
    else:
        invitation.role = role  # re-target an existing (possibly unused) invite
    db.commit()
    db.refresh(invitation)
    return invitation


def list_company_members(db: Session, company: Company, role: str) -> list[User]:
    """Return all users in ``company`` with the given ``role``."""
    return list(
        db.scalars(
            select(User)
            .join(UserProfile, UserProfile.user_id == User.id)
            .where(UserProfile.company_id == company.id, UserProfile.role == role)
        )
    )


def list_company_invitations(db: Session, company: Company) -> list[Invitation]:
    """Return invitations for ``company``, newest first."""
    return list(
        db.scalars(
            select(Invitation)
            .where(Invitation.company_id == company.id)
            .order_by(Invitation.created_at.desc())
        )
    )
