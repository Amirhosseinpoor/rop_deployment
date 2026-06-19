"""Password hashing and JWT helpers for the USAC service.

Kept separate from business logic so the crypto primitives are easy to audit and
swap. Uses ``passlib`` (bcrypt) for passwords and ``python-jose`` for JWTs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings

# bcrypt is a sensible, widely-supported default. ``deprecated="auto"`` lets us
# transparently upgrade hashes if the scheme list changes later.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return a salted bcrypt hash of ``plain``."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time check that ``plain`` matches the stored ``hashed`` value."""
    return _pwd_context.verify(plain, hashed)


def create_access_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    """Mint a signed JWT whose ``sub`` is ``subject`` (the username).

    Args:
        subject: Value placed in the ``sub`` claim (the user's username).
        extra_claims: Optional additional claims, e.g. ``{"role": "manager"}``.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_minutes),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate and decode a JWT, raising ``ValueError`` on any problem."""
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as e:  # signature/expiry/format errors
        raise ValueError(f"Invalid or expired token: {e}")
