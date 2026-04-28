# webapp/app/utils.py
import io
import base64
import datetime
import tempfile
import uuid

import torch
import torch.nn as nn
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b6
from django.core.files.base import ContentFile
from .rop_guidance import ROP_GUIDANCE
from segmentation_models_pytorch import UnetPlusPlus

# Allow-list target for PyTorch safe loader
try:
    from torchvision.models.efficientnet import EfficientNet
except Exception:
    EfficientNet = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globals
model_seg = None
model_plus = None
model_stage = None
model_zone = None
STAGE_ORDER = ["Stage 5", "Stage 4", "Stage 3", "Stage 2", "Stage 1", "Stage 0", "Normal"]
ZONE_ORDER  = ["Zone 1", "Zone 2", "Zone 3"]
PLUS_ORDER  = ["Plus", "No Plus"]
class_names = ["No Plus", "Plus"]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


def _severity_index(label: str, order: list[str]) -> int:
    try:
        return order.index(label)
    except ValueError:
        return len(order)

def decide_label(predictions: list[str], severity_order: list[str]) -> str:
    """
    Majority vote first; if tie, pick the worst according to severity_order.
    """
    counts = {}
    for p in predictions:
        counts[p] = counts.get(p, 0) + 1
    max_count = max(counts.values())
    max_labels = [lab for lab, c in counts.items() if c == max_count]
    if len(max_labels) == 1:
        return max_labels[0]
    for lab in severity_order:
        if lab in max_labels:
            return lab
    # fallback
    return max_labels[0]

def load_checkpoint_forgiving(model: nn.Module, path: str, device, strict: bool = False):
    try:
        if EfficientNet is not None:
            with torch.serialization.safe_globals([EfficientNet]):
                obj = torch.load(path, map_location=device, weights_only=True)
        else:
            try:
                dummy = efficientnet_b4(weights=None)
                with torch.serialization.safe_globals([type(dummy)]):
                    obj = torch.load(path, map_location=device, weights_only=True)
            except Exception:
                obj = torch.load(path, map_location=device, weights_only=True)
    except Exception as e_safe:
        try:
            obj = torch.load(path, map_location=device, weights_only=False)
        except Exception as e_unsafe:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}'.\n"
                f"Safe loader error: {e_safe}\n"
                f"Unsafe loader error: {e_unsafe}"
            )

    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "module", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
        else:
            state = obj

        if any(k.startswith("module.") for k in state.keys()):
            from collections import OrderedDict
            state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())

        model.load_state_dict(state, strict=strict)
        model = model.to(device).eval()
        return model

    raise RuntimeError(f"Unexpected checkpoint object type: {type(obj)} for '{path}'")


def get_segmentation_model():
    global model_seg
    if model_seg is None:
        seg = UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)
        weight_path = "model/best_weight_Unet++_maskresize_29"
        model_seg = load_checkpoint_forgiving(seg, weight_path, device, strict=False)
    return model_seg


def get_classification_model():
    global model_plus
    if model_plus is None:
        m = efficientnet_b4(weights=None).to(device)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1792, 2, bias=True)
        ).to(device)
        best_model_path = "model/model_efficentnet_b4_plus.pth"
        model_plus = load_checkpoint_forgiving(m, best_model_path, device, strict=False)
    return model_plus


def get_stage_model():
    global model_stage
    if model_stage is None:
        m = efficientnet_b6(weights=None).to(device)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(2304, 7, bias=True)
        ).to(device)
        best_model_path = "model/best_model (1).pth"
        model_stage = load_checkpoint_forgiving(m, best_model_path, device, strict=False)
    return model_stage


def get_zone_model():
    global model_zone
    if model_zone is None:
        m = efficientnet_b4(weights=None).to(device)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1792, 7, bias=True)
        ).to(device)
        zone_weights_path = "model/model_Zone_augment_Farabi_2"
        model_zone = load_checkpoint_forgiving(m, zone_weights_path, device, strict=False)
    return model_zone


def predict_mask(file_like, model, device, size=(512, 512)):
    file_bytes = np.asarray(bytearray(file_like.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from file object.")
    image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1)) / 255.0
    x = np.expand_dims(x, axis=0).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = (pred_y > 0.5).astype(np.uint8) * 255
        mask = np.stack([pred_y] * 3, axis=-1)
    return mask


def vessels(input_image_file, model, device):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_image.write(input_image_file.read())
        temp_image_path = temp_image.name

    original_image = cv2.imread(temp_image_path)
    with open(temp_image_path, "rb") as f:
        mask = predict_mask(f, model=model, device=device)

    cv2.imwrite(f'{temp_image_path}_mask.png', mask)

    vessel_image = cv2.imread(f"{temp_image_path}_mask.png", cv2.IMREAD_GRAYSCALE)
    vessel_image = cv2.resize(vessel_image, (original_image.shape[1], original_image.shape[0]))
    _, vessel_mask = cv2.threshold(vessel_image, 127, 255, cv2.THRESH_BINARY)

    result_image = original_image.copy()
    purple = np.array([128, 0, 128], dtype=np.uint8)
    alpha = 0.7
    result_image[vessel_mask == 255] = (
        alpha * purple + (1 - alpha) * result_image[vessel_mask == 255]
    ).astype(np.uint8)

    return result_image


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return simple_transform(image).unsqueeze(dim=0)


def get_prediction(image_bytes, model_plus):
    tensor = transform_image(image_bytes).to(device)
    with torch.inference_mode():
        outputs = model_plus(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    return class_names[pred_label], probs[0, pred_label].item()


def compute_final_decision(zone_label: str, stage_label: str, plus_label: str) -> str:
    """
    Decision system (from the attached flowchart):

    PLUS  ➜ Treatment

    NO PLUS:
      • Zone 1:
          - Stage 3   ➜ Treatment
          - Stage 1–2 ➜ Follow-up ≤ 1 week
          - No ROP    ➜ Follow-up 1–2 weeks
      • Zone 2:
          - Stage 3   ➜ Follow-up ≤ 1 week
          - Stage 2   ➜ Follow-up 1–2 weeks
          - Stage 1   ➜ Follow-up 2 weeks
          - No ROP    ➜ Follow-up 2–3 weeks
      • Zone 3:
          - Stage 1–2 ➜ Follow-up 2–3 weeks
    """

    # Normalize inputs
    z = zone_label.strip().lower().replace("zone", "").strip()
    s = stage_label.strip().lower().replace("stage", "").strip()
    p = plus_label.strip().lower()

    # PLUS: immediate treatment regardless of zone/stage
    if p == "plus":
        return "Treatment"

    # NO PLUS cases
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

    # Fallback if something doesn't match the diagram’s branches
    return "Follow-up"
def _norm(s: str) -> str:
    return s.strip().lower()

def get_guidance(zone_label: str, plus_label: str, stage_label: str, final_decision: str) -> dict:
    key = (_norm(zone_label), _norm(plus_label), _norm(stage_label))
    if key in ROP_GUIDANCE:
        return ROP_GUIDANCE[key]

    # Fallback: generic text from the decision tree if exact triple not found
    return {
        "title": f"{zone_label} · {plus_label} · {stage_label} → {final_decision}",
        "text": (
            f"Guidance not found in the static table for this exact combination. "
            f"Apply standard management for **{final_decision}** and follow local protocols."
        )
    }
def _predict_single_image(image_file, request=None):
    """
    Runs the full pipeline for ONE image and returns the same structure you already return in get_result,
    plus a few small extras to aid aggregation.
    """
    start_time = datetime.datetime.now()

    seg_model = get_segmentation_model()
    plus_model = get_classification_model()
    stage_model = get_stage_model()
    zone_model  = get_zone_model()

    # --- segmentation for Plus input ---
    image_file.seek(0)
    mask = predict_mask(image_file, seg_model, device)
    _, mask_buffer = cv2.imencode('.jpg', mask)
    mask_bytes = mask_buffer.tobytes()

    # --- Plus / No-Plus ---
    class_name, class_prob = get_prediction(mask_bytes, plus_model)

    # --- Stage ---
    image_file.seek(0)
    image_bytes = image_file.read()
    stage_tensor = transform_image(image_bytes).to(device)
    with torch.inference_mode():
        stage_output = stage_model(stage_tensor)
        stage_probs = torch.softmax(stage_output, dim=1)
        stage_label_idx = torch.argmax(stage_probs, dim=1).item()
    stage_names = ['Normal', 'Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5']
    stage_name = stage_names[stage_label_idx]
    stage_prob = stage_probs[0, stage_label_idx].item()

    # --- Zone (Z1/Z2 logits, else Z3) ---
    zone_tensor = transform_image(image_bytes).to(device)
    with torch.inference_mode():
        zone_logits_full = zone_model(zone_tensor)      # [1, 7]
        zone_logits = zone_logits_full[:, :2]
        zone_probs = torch.softmax(zone_logits, dim=1)
        max_prob, pred_12 = torch.max(zone_probs, dim=1)
    if max_prob.item() < 0.5:
        zone_label = "Zone 3"
        zone_conf = 1.0 - max_prob.item()
    else:
        zone_label = "Zone 1" if pred_12.item() == 0 else "Zone 2"
        zone_conf = max_prob.item()

    # --- Final decision ---
    final_decision = compute_final_decision(
        zone_label=zone_label,
        stage_label=stage_name,
        plus_label=class_name
    )

    # --- Guidance ---
    stage_key = stage_name if stage_name.lower().startswith("stage") else ("Normal" if stage_name.lower() == "normal" else stage_name)
    guidance = get_guidance(zone_label, class_name, stage_key, final_decision)

    # --- Vessel overlay for this image ---
    image_file.seek(0)
    segmented_image = vessels(image_file, seg_model, device)
    _, seg_buffer = cv2.imencode('.jpg', segmented_image)
    encoded_string = base64.b64encode(seg_buffer.tobytes()).decode('utf-8')
    overlay_data = f'data:image/jpeg;base64,{encoded_string}'
    original_data = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    end_time = datetime.datetime.now()
    execution_time = f'{round((end_time - start_time).total_seconds() * 1000)} ms'
    file_name = getattr(image_file, "name", "uploaded_image.jpg")

    diag_ctx = {
        "predictions": {"class_name": class_name, "class_prob": f"{class_prob:.3f}"},
        "stage_prediction": {"stage_name": stage_name, "stage_prob": f"{stage_prob:.3f}"},
        "zone_prediction": {"zone_name": zone_label, "zone_prob": f"{zone_conf:.3f}"},
        "final_decision": final_decision
    }
    diag_ctx_text = (
        f"Plus disease: {class_name} (p={class_prob:.3f}). "
        f"Stage: {stage_name} (p={stage_prob:.3f}). "
        f"Zone: {zone_label} (p={zone_conf:.3f}). "
        f"Final decision: {final_decision}."
    )

    # (Optional) logging as before — identical to your code
    image_url = None
    segmented_image_url = None
    if request is not None and hasattr(request, "user") and getattr(request.user, "is_authenticated", False):
        from .models import PredictionLog
        image_file.seek(0)
        log = PredictionLog.objects.create(
            user=request.user,
            file_name=file_name,
            predicted_class=class_name,
            probability=float(class_prob),
            stage_class=stage_name,
            stage_probability=float(stage_prob),
            zone_class=zone_label,
            zone_probability=float(zone_conf),
            final_decision=final_decision,
            execution_time=int(execution_time.replace(" ms", ""))
        )
        seg_image_name = f"segmented_{uuid.uuid4().hex}.jpg"
        seg_image_content = ContentFile(seg_buffer.tobytes(), name=seg_image_name)
        log.segmented_image.save(seg_image_name, seg_image_content)
        if hasattr(request, "build_absolute_uri"):
            log.segmented_image_url = request.build_absolute_uri(log.segmented_image.url)
            log.image = image_file
            log.save()
            log.image_url = request.build_absolute_uri(log.image.url)
            log.save(update_fields=["image_url", "segmented_image_url"])
        image_url = getattr(log, "image_url", None)
        segmented_image_url = getattr(log, "segmented_image_url", None)

    single = {
        "image_data": overlay_data,
        "original_image_data": original_data,
        "inference_time": execution_time,
        "file_name": file_name,
        "predictions": {"class_name": class_name, "class_prob": f"{class_prob:.3f}"},
        "stage_prediction": {"stage_name": stage_name, "stage_prob": f"{stage_prob:.3f}"},
        "zone_prediction": {"zone_name": zone_label, "zone_prob": f"{zone_conf:.3f}"},
        "final_decision": final_decision,
        "guidance": {"title": guidance["title"], "text": guidance["text"]},
        "diagnostic_context": diag_ctx,
        "diagnostic_context_text": diag_ctx_text,
        "llm_diagnostic_text": f"Plus: {class_name} (p={class_prob:.3f}); Stage: {stage_name} (p={stage_prob:.3f}); Zone: {zone_label} (p={zone_conf:.3f}); Final Decision: {final_decision}",
    }
    if image_url: single["image_url"] = image_url
    if segmented_image_url: single["segmented_image_url"] = segmented_image_url
    return single
def _score_for_worst(plus_label: str, stage_label: str, zone_label: str) -> tuple[int, int, int]:
    # lower tuple is worse
    return (
        _severity_index(plus_label, PLUS_ORDER),
        _severity_index(stage_label, STAGE_ORDER),
        _severity_index(zone_label, ZONE_ORDER),
    )

def get_results_for_images(image_files: list, request=None):
    """
    Predict each image; then compute the final (aggregated) labels:
      - Plus: majority → tie → worst by PLUS_ORDER
      - Zone: majority → tie → worst by ZONE_ORDER
      - Stage: majority → tie → worst by STAGE_ORDER
    Also choose one 'worst' image to visualize (based on (Plus, Stage, Zone) severity).
    """
    per_image = []
    for f in image_files:
        try:
            per_image.append(_predict_single_image(f, request=request))
        except Exception as e:
            # if one fails, keep going but mark error row
            per_image.append({"error": str(e), "file_name": getattr(f, "name", "file")})

    # Collect predictions where available
    plus_preds  = [r["predictions"]["class_name"] for r in per_image if "predictions" in r]
    stage_preds = [r["stage_prediction"]["stage_name"] for r in per_image if "stage_prediction" in r]
    zone_preds  = [r["zone_prediction"]["zone_name"] for r in per_image if "zone_prediction" in r]

    final_plus  = decide_label(plus_preds,  PLUS_ORDER)  if plus_preds  else "No Plus"
    final_stage = decide_label(stage_preds, STAGE_ORDER) if stage_preds else "Normal"
    final_zone  = decide_label(zone_preds,  ZONE_ORDER)  if zone_preds  else "Zone 3"

    # Choose one worst image to show
    worst_idx = None
    worst_key = None
    for idx, r in enumerate(per_image):
        if "predictions" not in r:  # skip errored
            continue
        k = _score_for_worst(
            r["predictions"]["class_name"],
            r["stage_prediction"]["stage_name"],
            r["zone_prediction"]["zone_name"]
        )
        if worst_key is None or k < worst_key:
            worst_key = k
            worst_idx = idx

    # fallbacks
    if worst_idx is None and per_image:
        worst_idx = 0

    worst_item = per_image[worst_idx] if (worst_idx is not None and worst_idx < len(per_image)) else None

    # Build an aggregated result (re-using your structure so the UI needs minimal change)
    aggregated = {
        "image_data": (worst_item or {}).get("image_data"),
        "original_image_data": (worst_item or {}).get("original_image_data"),
        "inference_time": (worst_item or {}).get("inference_time", ""),
        "file_name": (worst_item or {}).get("file_name", ""),
        "predictions": {"class_name": final_plus, "class_prob": (worst_item or {}).get("predictions", {}).get("class_prob", "")},
        "stage_prediction": {"stage_name": final_stage, "stage_prob": (worst_item or {}).get("stage_prediction", {}).get("stage_prob", "")},
        "zone_prediction": {"zone_name": final_zone, "zone_prob": (worst_item or {}).get("zone_prediction", {}).get("zone_prob", "")},
        "final_decision": compute_final_decision(zone_label=final_zone, stage_label=final_stage, plus_label=final_plus),
        "guidance": get_guidance(final_zone, final_plus, final_stage, compute_final_decision(final_zone, final_stage, final_plus)),
        "diagnostic_context": {
            "predictions": {"class_name": final_plus},
            "stage_prediction": {"stage_name": final_stage},
            "zone_prediction": {"zone_name": final_zone},
            "final_decision": compute_final_decision(final_zone, final_stage, final_plus)
        },
        "diagnostic_context_text": f"Plus disease: {final_plus}. Stage: {final_stage}. Zone: {final_zone}. Final decision: {compute_final_decision(final_zone, final_stage, final_plus)}.",
        "llm_diagnostic_text": f"Plus: {final_plus}; Stage: {final_stage}; Zone: {final_zone}; Final Decision: {compute_final_decision(final_zone, final_stage, final_plus)}",
        "worst_index": worst_idx,
    }

    return aggregated, per_image


def get_result(image_file, is_api=False, request=None):
    return _predict_single_image(image_file, request=request)
