"""Configuration for the USAC service.

Secrets and deployment details come exclusively from environment variables. The
JWT secret in particular MUST be overridden in production — the default is only a
developer convenience and is logged as a warning at import time.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

# A clearly-fake default so it is obvious when the secret hasn't been set.
_DEV_SECRET = "dev-only-insecure-change-me"


@dataclass(frozen=True)
class Settings:
    """Immutable configuration view."""

    # SQLAlchemy database URL. Defaults to a local SQLite file so the service is
    # runnable out-of-the-box; point it at Postgres in production.
    database_url: str = field(
        default_factory=lambda: os.getenv("USAC_DATABASE_URL", "sqlite:///./usac.db")
    )

    # JWT signing secret and algorithm.
    jwt_secret: str = field(default_factory=lambda: os.getenv("USAC_JWT_SECRET", _DEV_SECRET))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("USAC_JWT_ALGORITHM", "HS256"))

    # Access-token lifetime in minutes.
    access_token_minutes: int = field(
        default_factory=lambda: int(os.getenv("USAC_ACCESS_TOKEN_MINUTES", "720"))
    )

    # Network port. USAC owns 8003.
    port: int = field(default_factory=lambda: int(os.getenv("USAC_PORT", "8003")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    s = Settings()
    if s.jwt_secret == _DEV_SECRET:
        warnings.warn(
            "USAC_JWT_SECRET is using the insecure development default; "
            "set USAC_JWT_SECRET before deploying.",
            stacklevel=2,
        )
    return s
