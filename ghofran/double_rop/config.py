"""Runtime configuration for the Double-ROP service.

Everything that varies between deployments (model location, device, port) is an
environment variable so nothing is hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


# The checkpoint bundled in the ``model/`` directory next to this package,
# resolved absolutely so it is found regardless of the working directory.
_BUNDLED_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


@dataclass(frozen=True)
class Settings:
    """Immutable configuration view."""

    # Path to the trained EyeNet checkpoint (a plain state_dict).
    # Defaults to the bundled ``model/best_model.pth``; override with KC_MODEL_PATH
    # or MODEL_DIR (e.g. to point at the shared repo's model directory).
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "KC_MODEL_PATH",
            os.path.join(os.getenv("MODEL_DIR", _BUNDLED_MODEL_DIR), "best_model.pth"),
        )
    )

    # Torch device override; empty means auto-detect CUDA.
    device: str = field(default_factory=lambda: os.getenv("KC_DEVICE", ""))

    # Network port. Double-ROP owns 8002.
    port: int = field(default_factory=lambda: int(os.getenv("KC_PORT", "8002")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (configuration is read once)."""
    return Settings()
