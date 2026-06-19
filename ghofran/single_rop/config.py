"""Runtime configuration for the Single-ROP service.

All values are read from environment variables so that *nothing* about the
deployment (model locations, network port, device) is hard-coded. This keeps
the service portable and satisfies the "no hardcoded configuration/secrets"
requirement.

Why a dedicated module instead of reading ``os.environ`` inline:
- a single, documented place to discover every knob the service exposes;
- defaults live next to their documentation;
- importing ``settings`` gives editors/type-checkers concrete attributes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


# The four model checkpoints the pipeline needs. The original Django code used
# relative paths rooted at the project (e.g. "model/best_model (1).pth"). We keep
# the same *file names* but resolve them under a configurable directory so the
# service can find the weights regardless of the working directory.
_DEFAULT_WEIGHTS = {
    "segmentation": "best_weight_Unet++_maskresize_29",
    "plus": "model_efficentnet_b4_plus.pth",
    "stage": "best_model (1).pth",
    "zone": "model_Zone_augment_Farabi_2",
}

# Default weights live in the ``model/`` directory bundled next to this package,
# resolved absolutely so the service works no matter the current directory.
# ``MODEL_DIR`` env var still overrides it (e.g. to point at the shared repo).
_BUNDLED_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


@dataclass(frozen=True)
class Settings:
    """Immutable view of the service configuration."""

    # Directory that holds the four PyTorch checkpoints listed in _DEFAULT_WEIGHTS.
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", _BUNDLED_MODEL_DIR))

    # Torch device override. When empty we auto-detect CUDA at runtime.
    device: str = field(default_factory=lambda: os.getenv("ROP_DEVICE", ""))

    # Network port the ASGI server should bind to. Single-ROP owns 8001.
    port: int = field(default_factory=lambda: int(os.getenv("ROP_PORT", "8001")))

    # Decision threshold for the zone head: below it we fall back to "Zone 3".
    zone_threshold: float = field(
        default_factory=lambda: float(os.getenv("ROP_ZONE_THRESHOLD", "0.5"))
    )

    def weight_path(self, key: str) -> str:
        """Absolute path of a checkpoint identified by its logical ``key``."""
        return os.path.join(self.model_dir, _DEFAULT_WEIGHTS[key])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    Cached because configuration is read once at start-up and never changes
    during the lifetime of the process.
    """
    return Settings()
