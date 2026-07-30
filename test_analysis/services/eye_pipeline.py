"""
Two-phase eye-segmentation + anemia-classification pipeline.

Mirrors the training notebooks (ipy/eye-defy-phase1-segmentation.ipynb and
ipy/phase2.ipynb):

  Phase 1  Simple U-Net (smp.Unet, resnet34 backbone, 1 class) segments the
           forniceal+palpebral conjunctiva from the raw eye photo.  Weights:
           model/all_models_weights/best_Simple_UNet.pth
  Phase 2  Same architecture segments the palpebral region from the phase-1 RGB
           crop (background blacked out).  Weights:
           model/phase2_palpebral_weights_kfold/phase2_Simple_UNet_fold0.pth
           (kept in place, but its output is NOT what the classifier consumes)
  Classify Anemia positive/negative — real EfficientNet-B0 classifier trained on
           forniceal_palpebral crops (see ipy/load_and_infer.py).  It runs on the
           PHASE-1 RGB crop (the forniceal+palpebral region), not the phase-2 crop.
           Weights: model/model_weigh/best_efficientnet_b0_forniceal_palpebral.pth

Both models are lazily built once and cached.  Pre/post-processing replicates the
notebooks' val transform: BGR→RGB, resize 256×256, ImageNet normalize, ToTensor,
sigmoid > 0.5.  Albumentations is not required — the transform is reproduced with
cv2/numpy so the app has no extra dependency.
"""
from __future__ import annotations

import io
import logging
import os

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)

# ---- configuration --------------------------------------------------------
IMG_SIZE = 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PHASE1_WEIGHTS = os.path.join(settings.BASE_DIR, "model", "all_models_weights", "best_Simple_UNet.pth")
PHASE2_WEIGHTS = os.path.join(
    settings.BASE_DIR, "model", "phase2_palpebral_weights_kfold", "phase2_Simple_UNet_fold0.pth"
)
# Best overall anemia classifier (F1/Acc/AUC all 0.903): EfficientNet-B0 trained on
# forniceal_palpebral crops — i.e. exactly the phase-1 conjunctiva crop.
CLASSIFIER_WEIGHTS = os.path.join(
    settings.BASE_DIR, "model", "model_weigh", "best_efficientnet_b0_forniceal_palpebral.pth"
)

# Cached singletons (model, device) so we build/load weights only once.
_CACHE: dict = {}


def _get_model(weights_path: str, key: str):
    """Build a Simple U-Net and load the given weights (cached per key)."""
    if key in _CACHE:
        return _CACHE[key]

    import torch
    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    logger.info("🧠 loaded eye-segmentation weights: %s (device=%s)", os.path.basename(weights_path), device)
    _CACHE[key] = (model, device)
    return _CACHE[key]


def _get_classifier():
    """Build the EfficientNet-B0 anemia classifier and load its weights (cached).

    Architecture + weight loading mirror ipy/load_and_infer.py: a torchvision
    efficientnet_b0 with the final classifier layer replaced by a single-logit
    Linear head, loaded strict=True from the forniceal_palpebral checkpoint.
    """
    key = "classifier"
    if key in _CACHE:
        return _CACHE[key]

    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    state = torch.load(CLASSIFIER_WEIGHTS, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    logger.info("🧠 loaded anemia classifier: %s (device=%s)", os.path.basename(CLASSIFIER_WEIGHTS), device)
    _CACHE[key] = (model, device)
    return _CACHE[key]


# ---- image helpers --------------------------------------------------------
def _to_tensor(rgb: np.ndarray):
    """RGB uint8 (H,W,3) → normalized float tensor (1,3,256,256)."""
    import torch

    img = _cv2().resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=_cv2().INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return torch.from_numpy(img).unsqueeze(0)


def _cv2():
    import cv2
    return cv2


def _predict_mask(model, device, rgb: np.ndarray) -> np.ndarray:
    """Run one segmentation phase; return a full-resolution binary mask (H,W) uint8 {0,1}."""
    import torch

    h, w = rgb.shape[:2]
    x = _to_tensor(rgb).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (256,256)
    mask_small = (prob > 0.5).astype(np.uint8)
    mask = _cv2().resize(mask_small, (w, h), interpolation=_cv2().INTER_NEAREST)
    return mask


def _apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep pixels where mask==1, black elsewhere (matches the notebooks' RGB crop)."""
    return rgb * mask[:, :, None]


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 array to PNG bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _mask_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a binary mask {0,1} to a grayscale PNG (0/255)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _classify_anemia(phase1_rgb: np.ndarray) -> tuple[str, float]:
    """Real anemia classifier (EfficientNet-B0, forniceal_palpebral).

    Input is the PHASE-1 RGB crop — the forniceal+palpebral conjunctiva isolated
    against a black background — which is exactly the crop distribution the model
    was trained on.  Preprocessing (resize 256×256 + ImageNet normalize) matches
    ipy/load_and_infer.py; a single output logit → sigmoid → P(anemic), threshold
    0.5.  Returns ('positive'|'negative', confidence-in-the-chosen-label 0..1).
    """
    import torch

    model, device = _get_classifier()
    x = _to_tensor(phase1_rgb).to(device)
    with torch.no_grad():
        logit = model(x).squeeze()
        prob_anemic = float(torch.sigmoid(logit).item())
    if prob_anemic > 0.5:
        return "positive", round(prob_anemic, 3)
    return "negative", round(1.0 - prob_anemic, 3)


# ---- public entry point ---------------------------------------------------
def analyze_eye_image(eye_image) -> "object":
    """Run the full pipeline for one EyeImage and persist an EyeAnalysis row.

    Best-effort: never raises — on failure it records status='failed' with the
    error so the caller (a web request) is never broken by model issues.
    Returns the EyeAnalysis instance.
    """
    from test_analysis.models import EyeAnalysis

    analysis, _ = EyeAnalysis.objects.get_or_create(eye_image=eye_image)
    try:
        cv2 = _cv2()
        # 1. Load the original photo as RGB.
        bgr = cv2.imread(eye_image.image.path)
        if bgr is None:
            raise ValueError(f"could not read image: {eye_image.image.path}")
        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 2. Phase 1 — forniceal+palpebral segmentation.
        m1, d1 = _get_model(PHASE1_WEIGHTS, "phase1")
        mask1 = _predict_mask(m1, d1, original_rgb)
        phase1_rgb = _apply_mask(original_rgb, mask1)

        # 3. Phase 2 — palpebral segmentation from the phase-1 RGB crop.
        #    Kept in place for display, but the classifier does NOT consume it.
        m2, d2 = _get_model(PHASE2_WEIGHTS, "phase2")
        mask2 = _predict_mask(m2, d2, phase1_rgb)
        phase2_rgb = _apply_mask(phase1_rgb, mask2)

        # 4. Real anemia classification from the PHASE-1 forniceal_palpebral crop.
        label, confidence = _classify_anemia(phase1_rgb)

        # 5. Persist derived images + result.
        stem = f"eye{eye_image.id}"
        analysis.phase1_mask.save(f"{stem}_p1_mask.png", ContentFile(_mask_png_bytes(mask1)), save=False)
        analysis.phase1_overlay.save(f"{stem}_p1_rgb.png", ContentFile(_png_bytes(phase1_rgb)), save=False)
        analysis.phase2_mask.save(f"{stem}_p2_mask.png", ContentFile(_mask_png_bytes(mask2)), save=False)
        analysis.phase2_overlay.save(f"{stem}_p2_rgb.png", ContentFile(_png_bytes(phase2_rgb)), save=False)
        analysis.anemia_label = label
        analysis.anemia_confidence = confidence
        analysis.status = EyeAnalysis.STATUS_DONE
        analysis.error = None
        analysis.save()
        logger.info("👁️ eye analysis done | eye_image=%s | anemia=%s (%.2f)", eye_image.id, label, confidence)
    except Exception as e:  # noqa: BLE001 - must never break the caller
        logger.exception("🔥 eye analysis failed for eye_image=%s: %s", getattr(eye_image, "id", None), e)
        analysis.status = EyeAnalysis.STATUS_FAILED
        analysis.error = str(e)
        analysis.save()
    return analysis
