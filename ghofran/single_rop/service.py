"""Core ROP inference logic — completely free of any web framework.

This module is a faithful port of the original Django ``single_rop/utils.py``
pipeline. The clinical algorithm (segmentation -> plus/stage/zone heads ->
decision table -> guidance lookup) is preserved exactly; what changed is:

* **No Django.** The original optionally wrote a ``PredictionLog`` row and saved
  images to ``MEDIA_ROOT``. That coupled inference to a database and a request
  object. A microservice should do one thing well, so persistence was removed.
  Images are returned to the caller as base64 data-URIs (already produced by the
  original code) instead of being written to disk.
* **Configurable model paths.** Weight locations come from :mod:`config` rather
  than being hard-coded relative strings.
* **Lazy, cached model loading.** Identical behaviour to the original globals,
  but expressed through a small registry so it is easy to reason about.

The functions accept raw image *bytes* rather than Django ``UploadedFile``
objects, which keeps the logic decoupled from HTTP entirely (the route layer is
responsible for turning an upload into bytes).
"""
from __future__ import annotations

import base64
import datetime
import io
import tempfile
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from segmentation_models_pytorch import UnetPlusPlus
from torchvision import transforms
from torchvision.models import efficientnet_b4, efficientnet_b6

from .config import get_settings
from .rop_guidance import ROP_GUIDANCE

# ``EfficientNet`` is allow-listed for PyTorch's "safe" checkpoint loader. Some
# checkpoints in this project were pickled as whole modules, so the safe loader
# needs the class registered. Import is best-effort to stay forward-compatible.
try:  # pragma: no cover - depends on torchvision version
    from torchvision.models.efficientnet import EfficientNet
except Exception:  # noqa: BLE001 - any import failure means "not available"
    EfficientNet = None


# --------------------------------------------------------------------------- #
# Constants — copied verbatim from the original pipeline so results are identical
# --------------------------------------------------------------------------- #
# Severity orderings are "worst first": index 0 is the most severe label. They
# drive both tie-breaking in majority voting and the "pick the worst image"
# logic when several fundus images are analysed together.
STAGE_ORDER = ["Stage 5", "Stage 4", "Stage 3", "Stage 2", "Stage 1", "Stage 0", "Normal"]
ZONE_ORDER = ["Zone 1", "Zone 2", "Zone 3"]
PLUS_ORDER = ["Plus", "No Plus"]
CLASS_NAMES = ["No Plus", "Plus"]  # index order of the plus-disease head
STAGE_NAMES = ["Normal", "Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]

# ImageNet normalisation — the models were trained with these statistics.
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
_SIMPLE_TRANSFORM = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), _NORMALIZE]
)


def _resolve_device() -> torch.device:
    """Pick the torch device: explicit override, else CUDA when present."""
    override = get_settings().device
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Module-level device handle (matches the original module-global behaviour).
device = _resolve_device()


# --------------------------------------------------------------------------- #
# Model registry — lazy singletons, loaded on first use and then re-used.
# --------------------------------------------------------------------------- #
_models: dict[str, nn.Module] = {}


def _load_checkpoint_forgiving(
    model: nn.Module, path: str, strict: bool = False
) -> nn.Module:
    """Load weights from ``path`` tolerating the many ways they were saved.

    Checkpoints in this project come in several shapes: whole pickled modules,
    plain ``state_dict``s, and dicts wrapping a state under various keys, some
    with a ``module.`` prefix from ``DataParallel`` training. This helper tries
    the safe (``weights_only``) loader first and falls back to the unsafe loader,
    mirroring the original implementation so existing weights keep loading.
    """
    try:
        if EfficientNet is not None:
            with torch.serialization.safe_globals([EfficientNet]):
                obj = torch.load(path, map_location=device, weights_only=True)
        else:
            obj = torch.load(path, map_location=device, weights_only=True)
    except Exception as e_safe:  # noqa: BLE001
        try:
            obj = torch.load(path, map_location=device, weights_only=False)
        except Exception as e_unsafe:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load checkpoint '{path}'.\n"
                f"Safe loader error: {e_safe}\n"
                f"Unsafe loader error: {e_unsafe}"
            )

    # Case 1: the checkpoint *is* a ready-to-use module.
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()

    # Case 2: a dict — unwrap the state_dict from the usual wrapper keys.
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "module", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
        else:
            state = obj

        if any(k.startswith("module.") for k in state.keys()):
            state = OrderedDict(
                (k.replace("module.", "", 1), v) for k, v in state.items()
            )

        model.load_state_dict(state, strict=strict)
        return model.to(device).eval()

    raise RuntimeError(f"Unexpected checkpoint object type: {type(obj)} for '{path}'")


def _get_segmentation_model() -> nn.Module:
    """Unet++ (ResNet18 encoder) that isolates the retinal vasculature."""
    if "seg" not in _models:
        seg = UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(device)
        _models["seg"] = _load_checkpoint_forgiving(
            seg, get_settings().weight_path("segmentation")
        )
    return _models["seg"]


def _get_plus_model() -> nn.Module:
    """EfficientNet-B4 head that classifies Plus / No-Plus disease."""
    if "plus" not in _models:
        m = efficientnet_b4(weights=None).to(device)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(1792, 2, bias=True)
        ).to(device)
        _models["plus"] = _load_checkpoint_forgiving(
            m, get_settings().weight_path("plus")
        )
    return _models["plus"]


def _get_stage_model() -> nn.Module:
    """EfficientNet-B6 head that predicts disease stage (7 classes)."""
    if "stage" not in _models:
        m = efficientnet_b6(weights=None).to(device)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(2304, 7, bias=True)
        ).to(device)
        _models["stage"] = _load_checkpoint_forgiving(
            m, get_settings().weight_path("stage")
        )
    return _models["stage"]


def _get_zone_model() -> nn.Module:
    """EfficientNet-B4 head whose first two logits encode Zone 1 vs Zone 2."""
    if "zone" not in _models:
        m = efficientnet_b4(weights=None).to(device)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(1792, 7, bias=True)
        ).to(device)
        _models["zone"] = _load_checkpoint_forgiving(
            m, get_settings().weight_path("zone")
        )
    return _models["zone"]


def warmup_models() -> None:
    """Eagerly load every model. Useful to call at start-up so the first
    request does not pay the (large) one-time loading cost."""
    _get_segmentation_model()
    _get_plus_model()
    _get_stage_model()
    _get_zone_model()


# --------------------------------------------------------------------------- #
# Small pure helpers (voting / severity) — straight ports.
# --------------------------------------------------------------------------- #
def _severity_index(label: str, order: list[str]) -> int:
    """Position of ``label`` in a worst-first ordering (missing -> least severe)."""
    try:
        return order.index(label)
    except ValueError:
        return len(order)


def decide_label(predictions: list[str], severity_order: list[str]) -> str:
    """Majority vote across ``predictions``; ties broken toward the worst label."""
    counts: dict[str, int] = {}
    for p in predictions:
        counts[p] = counts.get(p, 0) + 1
    max_count = max(counts.values())
    max_labels = [lab for lab, c in counts.items() if c == max_count]
    if len(max_labels) == 1:
        return max_labels[0]
    for lab in severity_order:  # worst-first => first match is the worst tied label
        if lab in max_labels:
            return lab
    return max_labels[0]


def _score_for_worst(plus_label: str, stage_label: str, zone_label: str) -> tuple:
    """Tuple where a *smaller* value means a *worse* (more severe) image."""
    return (
        _severity_index(plus_label, PLUS_ORDER),
        _severity_index(stage_label, STAGE_ORDER),
        _severity_index(zone_label, ZONE_ORDER),
    )


# --------------------------------------------------------------------------- #
# Image / tensor utilities.
# --------------------------------------------------------------------------- #
def _predict_mask(image_bytes: bytes, model: nn.Module, size=(512, 512)) -> np.ndarray:
    """Run the segmentation model and return a 3-channel binary vessel mask."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes.")
    image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1)) / 255.0
    x = np.expand_dims(x, axis=0).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0].cpu().numpy()
        pred = np.squeeze(pred, axis=0)
        pred = (pred > 0.5).astype(np.uint8) * 255
        return np.stack([pred] * 3, axis=-1)


def _vessel_overlay(image_bytes: bytes, model: nn.Module) -> np.ndarray:
    """Overlay the predicted vasculature on the original image in purple."""
    # cv2 needs a file path for some legacy reads; a temp file keeps parity with
    # the original implementation and handles arbitrary input encodings.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    original_image = cv2.imread(tmp_path)
    mask = _predict_mask(image_bytes, model)
    mask_path = f"{tmp_path}_mask.png"
    cv2.imwrite(mask_path, mask)

    vessel = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    vessel = cv2.resize(vessel, (original_image.shape[1], original_image.shape[0]))
    _, vessel_mask = cv2.threshold(vessel, 127, 255, cv2.THRESH_BINARY)

    result = original_image.copy()
    purple = np.array([128, 0, 128], dtype=np.uint8)
    alpha = 0.7
    result[vessel_mask == 255] = (
        alpha * purple + (1 - alpha) * result[vessel_mask == 255]
    ).astype(np.uint8)
    return result


def _transform_image(image_bytes: bytes) -> torch.Tensor:
    """Decode bytes to a normalised (1, 3, 224, 224) tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _SIMPLE_TRANSFORM(image).unsqueeze(dim=0)


def _get_plus_prediction(mask_bytes: bytes, model: nn.Module) -> tuple[str, float]:
    """Classify Plus/No-Plus from the segmented vessel mask."""
    tensor = _transform_image(mask_bytes).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)
        idx = torch.argmax(probs, dim=1).item()
    return CLASS_NAMES[idx], probs[0, idx].item()


# --------------------------------------------------------------------------- #
# Clinical decision + guidance — verbatim clinical logic.
# --------------------------------------------------------------------------- #
def compute_final_decision(zone_label: str, stage_label: str, plus_label: str) -> str:
    """Aggregate (Zone, Stage, Plus) into a management recommendation.

    Encodes the ICROP/ETROP decision flowchart from the original system: Plus
    disease always means "Treatment"; otherwise the follow-up interval depends
    on the zone/stage combination.
    """
    z = zone_label.strip().lower().replace("zone", "").strip()
    s = stage_label.strip().lower().replace("stage", "").strip()
    p = plus_label.strip().lower()

    if p == "plus":
        return "Treatment"

    if z == "1":
        if s == "3":
            return "Treatment"
        if s in {"1-2", "1–2", "1", "2"}:
            return "Follow-up ≤ 1 week"
        if stage_label.strip().lower() in {"no rop", "norop"}:
            return "Follow-up 1–2 weeks"
    elif z == "2":
        if s == "3":
            return "Follow-up ≤ 1 week"
        if s == "2":
            return "Follow-up 1–2 weeks"
        if s == "1":
            return "Follow-up 2 weeks"
        if stage_label.strip().lower() in {"no rop", "norop"}:
            return "Follow-up 2–3 weeks"
    elif z == "3":
        if s in {"1-2", "1–2", "1", "2"}:
            return "Follow-up 2–3 weeks"

    return "Follow-up"


def _norm(s: str) -> str:
    return s.strip().lower()


def get_guidance(
    zone_label: str, plus_label: str, stage_label: str, final_decision: str
) -> dict:
    """Look up the long-form clinical guidance for a (zone, plus, stage) triple."""
    key = (_norm(zone_label), _norm(plus_label), _norm(stage_label))
    if key in ROP_GUIDANCE:
        return ROP_GUIDANCE[key]
    return {
        "title": f"{zone_label} · {plus_label} · {stage_label} → {final_decision}",
        "text": (
            "Guidance not found in the static table for this exact combination. "
            f"Apply standard management for **{final_decision}** and follow local protocols."
        ),
    }


# --------------------------------------------------------------------------- #
# Public entry points.
# --------------------------------------------------------------------------- #
def predict_single_image(image_bytes: bytes, file_name: str = "uploaded_image.jpg") -> dict:
    """Run the full pipeline for ONE image and return the result dictionary.

    Args:
        image_bytes: Raw bytes of a fundus image (JPEG/PNG/...).
        file_name: Original file name, echoed back for the caller's convenience.

    Returns:
        A dict containing per-head predictions, the aggregate decision, the
        clinical guidance text, and base64 data-URIs for the original image and
        the vessel overlay.
    """
    start_time = datetime.datetime.now()

    seg_model = _get_segmentation_model()
    plus_model = _get_plus_model()
    stage_model = _get_stage_model()
    zone_model = _get_zone_model()

    # --- Segmentation -> Plus/No-Plus ---
    mask = _predict_mask(image_bytes, seg_model)
    _, mask_buffer = cv2.imencode(".jpg", mask)
    class_name, class_prob = _get_plus_prediction(mask_buffer.tobytes(), plus_model)

    # --- Stage ---
    stage_tensor = _transform_image(image_bytes).to(device)
    with torch.inference_mode():
        stage_probs = torch.softmax(stage_model(stage_tensor), dim=1)
        stage_idx = torch.argmax(stage_probs, dim=1).item()
    stage_name = STAGE_NAMES[stage_idx]
    stage_prob = stage_probs[0, stage_idx].item()

    # --- Zone (first two logits = Z1/Z2, else fall back to Z3) ---
    zone_tensor = _transform_image(image_bytes).to(device)
    with torch.inference_mode():
        zone_logits = zone_model(zone_tensor)[:, :2]
        zone_probs = torch.softmax(zone_logits, dim=1)
        max_prob, pred_12 = torch.max(zone_probs, dim=1)
    if max_prob.item() < get_settings().zone_threshold:
        zone_label = "Zone 3"
        zone_conf = 1.0 - max_prob.item()
    else:
        zone_label = "Zone 1" if pred_12.item() == 0 else "Zone 2"
        zone_conf = max_prob.item()

    # --- Decision + guidance ---
    final_decision = compute_final_decision(zone_label, stage_name, class_name)
    guidance = get_guidance(zone_label, class_name, stage_name, final_decision)

    # --- Vessel overlay + base64 images ---
    overlay = _vessel_overlay(image_bytes, seg_model)
    _, seg_buffer = cv2.imencode(".jpg", overlay)
    overlay_data = "data:image/jpeg;base64," + base64.b64encode(seg_buffer.tobytes()).decode("utf-8")
    original_data = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

    execution_ms = round((datetime.datetime.now() - start_time).total_seconds() * 1000)

    diag_ctx = {
        "predictions": {"class_name": class_name, "class_prob": f"{class_prob:.3f}"},
        "stage_prediction": {"stage_name": stage_name, "stage_prob": f"{stage_prob:.3f}"},
        "zone_prediction": {"zone_name": zone_label, "zone_prob": f"{zone_conf:.3f}"},
        "final_decision": final_decision,
    }
    diag_ctx_text = (
        f"Plus disease: {class_name} (p={class_prob:.3f}). "
        f"Stage: {stage_name} (p={stage_prob:.3f}). "
        f"Zone: {zone_label} (p={zone_conf:.3f}). "
        f"Final decision: {final_decision}."
    )

    return {
        "image_data": overlay_data,
        "original_image_data": original_data,
        "inference_time": f"{execution_ms} ms",
        "file_name": file_name,
        "predictions": {"class_name": class_name, "class_prob": f"{class_prob:.3f}"},
        "stage_prediction": {"stage_name": stage_name, "stage_prob": f"{stage_prob:.3f}"},
        "zone_prediction": {"zone_name": zone_label, "zone_prob": f"{zone_conf:.3f}"},
        "final_decision": final_decision,
        "guidance": {"title": guidance["title"], "text": guidance["text"]},
        "diagnostic_context": diag_ctx,
        "diagnostic_context_text": diag_ctx_text,
        "llm_diagnostic_text": (
            f"Plus: {class_name} (p={class_prob:.3f}); "
            f"Stage: {stage_name} (p={stage_prob:.3f}); "
            f"Zone: {zone_label} (p={zone_conf:.3f}); "
            f"Final Decision: {final_decision}"
        ),
    }


def predict_many_images(images: list[tuple[bytes, str]]) -> tuple[dict, list[dict]]:
    """Analyse several fundus images and aggregate them into a final verdict.

    Args:
        images: list of ``(image_bytes, file_name)`` tuples.

    Returns:
        ``(aggregated, per_image)`` where ``per_image`` holds one result dict per
        input (or an ``{"error": ...}`` marker if that image failed), and
        ``aggregated`` carries the majority-voted labels plus the visualisation
        of the single worst image.
    """
    per_image: list[dict] = []
    for image_bytes, name in images:
        try:
            per_image.append(predict_single_image(image_bytes, file_name=name))
        except Exception as e:  # noqa: BLE001 - one bad image shouldn't abort the batch
            per_image.append({"error": str(e), "file_name": name})

    plus_preds = [r["predictions"]["class_name"] for r in per_image if "predictions" in r]
    stage_preds = [r["stage_prediction"]["stage_name"] for r in per_image if "stage_prediction" in r]
    zone_preds = [r["zone_prediction"]["zone_name"] for r in per_image if "zone_prediction" in r]

    final_plus = decide_label(plus_preds, PLUS_ORDER) if plus_preds else "No Plus"
    final_stage = decide_label(stage_preds, STAGE_ORDER) if stage_preds else "Normal"
    final_zone = decide_label(zone_preds, ZONE_ORDER) if zone_preds else "Zone 3"

    # Pick one "worst" image to visualise in the aggregate view.
    worst_idx: Optional[int] = None
    worst_key = None
    for idx, r in enumerate(per_image):
        if "predictions" not in r:
            continue
        k = _score_for_worst(
            r["predictions"]["class_name"],
            r["stage_prediction"]["stage_name"],
            r["zone_prediction"]["zone_name"],
        )
        if worst_key is None or k < worst_key:
            worst_key, worst_idx = k, idx
    if worst_idx is None and per_image:
        worst_idx = 0
    worst = per_image[worst_idx] if (worst_idx is not None and worst_idx < len(per_image)) else {}

    final_dec = compute_final_decision(final_zone, final_stage, final_plus)
    aggregated = {
        "image_data": worst.get("image_data"),
        "original_image_data": worst.get("original_image_data"),
        "inference_time": worst.get("inference_time", ""),
        "file_name": worst.get("file_name", ""),
        "predictions": {"class_name": final_plus, "class_prob": worst.get("predictions", {}).get("class_prob", "")},
        "stage_prediction": {"stage_name": final_stage, "stage_prob": worst.get("stage_prediction", {}).get("stage_prob", "")},
        "zone_prediction": {"zone_name": final_zone, "zone_prob": worst.get("zone_prediction", {}).get("zone_prob", "")},
        "final_decision": final_dec,
        "guidance": get_guidance(final_zone, final_plus, final_stage, final_dec),
        "diagnostic_context": {
            "predictions": {"class_name": final_plus},
            "stage_prediction": {"stage_name": final_stage},
            "zone_prediction": {"zone_name": final_zone},
            "final_decision": final_dec,
        },
        "diagnostic_context_text": (
            f"Plus disease: {final_plus}. Stage: {final_stage}. "
            f"Zone: {final_zone}. Final decision: {final_dec}."
        ),
        "llm_diagnostic_text": (
            f"Plus: {final_plus}; Stage: {final_stage}; "
            f"Zone: {final_zone}; Final Decision: {final_dec}"
        ),
        "worst_index": worst_idx,
    }
    return aggregated, per_image
