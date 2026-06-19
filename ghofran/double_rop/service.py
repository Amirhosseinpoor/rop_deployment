"""Framework-free inference logic for the Double-ROP service.

Ported from the original ``double_rop/utils.py``. Differences:

* **No Django / DB.** The original optionally persisted a ``PredictionResult``
  row and request-built absolute URLs. A microservice shouldn't own a database,
  so persistence was dropped; base64 previews of the inputs are returned instead.
* **Lazy model loading.** The original loaded the model at import time, which
  makes the module impossible to import without the weights present. Here the
  model is loaded on first use and cached, so the package imports cleanly and the
  weights are only needed when a prediction is actually requested.
* **Configurable paths/device** via :mod:`config`.
"""
from __future__ import annotations

import base64
import io

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import get_settings
from .model import EyeNet, config

# The model expects 224x224 RGB tensors in [0, 1] (no normalisation in the
# original pipeline — kept identical here so results match).
_SIMPLE_TRANSFORM = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

# Cached singletons, populated on first use.
_device: torch.device | None = None
_model: EyeNet | None = None


def _resolve_device() -> torch.device:
    override = get_settings().device
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    """Return (and memoise) the torch device for this process."""
    global _device
    if _device is None:
        _device = _resolve_device()
    return _device


def get_model() -> EyeNet:
    """Load the EyeNet model once and cache it.

    The checkpoint is a plain ``state_dict`` saved from a ``resnet50`` EyeNet, so
    we instantiate the matching architecture and load weights with the safe
    (``weights_only``) loader.
    """
    global _model
    if _model is None:
        device = get_device()
        model = EyeNet().to(device)
        state = torch.load(
            get_settings().model_path, map_location=device, weights_only=True
        )
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model


def warmup_model() -> None:
    """Eagerly load the model so the first request isn't slow."""
    get_model()


def _process_image(image_bytes: bytes) -> torch.Tensor:
    """Decode bytes into a model-ready (1, 3, 224, 224) tensor on the device."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Image processing failed: {e}")
    return _SIMPLE_TRANSFORM(image).unsqueeze(0).to(get_device())


def _encode_image(image_bytes: bytes) -> str:
    """Build a base64 PNG data-URI for web display."""
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"


def get_prediction(left_bytes: bytes, right_bytes: bytes) -> dict:
    """Run the binocular classifier on a left/right image pair.

    Args:
        left_bytes: Raw bytes of the left-eye image.
        right_bytes: Raw bytes of the right-eye image.

    Returns:
        Dict with ``left_eye``, ``right_eye`` and ``z_class`` (each a
        ``{label, probability}`` pair) plus base64 previews under ``image_data``.
    """
    try:
        model = get_model()
        left_tensor = _process_image(left_bytes)
        right_tensor = _process_image(right_bytes)

        with torch.no_grad():
            left_out, right_out, z_out = model(left_tensor, right_tensor)

        left_probs = F.softmax(left_out, dim=1).squeeze()
        right_probs = F.softmax(right_out, dim=1).squeeze()
        z_probs = F.softmax(z_out, dim=1).squeeze()

        return {
            "left_eye": {
                "label": config.classes_xy[torch.argmax(left_probs).item()],
                "probability": f"{torch.max(left_probs).item():.4f}",
            },
            "right_eye": {
                "label": config.classes_xy[torch.argmax(right_probs).item()],
                "probability": f"{torch.max(right_probs).item():.4f}",
            },
            "z_class": {
                "label": config.classes_z[torch.argmax(z_probs).item()],
                "probability": f"{torch.max(z_probs).item():.4f}",
            },
            "image_data": {
                "left": _encode_image(left_bytes),
                "right": _encode_image(right_bytes),
            },
        }
    except ValueError:
        raise  # bad input — let the route turn it into a 400
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Prediction failed: {e}")
