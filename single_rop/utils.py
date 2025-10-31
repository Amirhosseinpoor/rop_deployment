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

class_names = ["No Plus", "Plus"]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


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
    if stage_label in {"Stage 4", "Stage 5"}:
        return "Treatment"

    if zone_label == "Zone 1":
        if plus_label == "Plus":
            return "Treatment"
        else:
            if stage_label == "Stage 3":
                return "Treatment"
            if stage_label in {"Stage 1", "Stage 2"}:
                return "Follow-up ≤ 1 week"
            if stage_label in {"Normal", "Stage 0"}:
                return "Follow-up 1–2 weeks"

    if zone_label == "Zone 2":
        if plus_label == "Plus":
            return "Treatment"
        else:
            if stage_label == "Stage 3":
                return "Follow-up ≤ 1 week"
            if stage_label == "Stage 2":
                return "Follow-up 1–2 weeks"
            if stage_label == "Stage 1":
                return "Follow-up 2 weeks"
            if stage_label in {"Normal", "Stage 0"}:
                return "Follow-up 2–3 weeks"

    if zone_label == "Zone 3":
        return "Follow-up 2–3 weeks"

    return "Follow-up"


def get_result(image_file, is_api=False, request=None):
    """
    Runs segmentation -> Plus classification -> Stage -> Zone -> Final decision,
    builds a unified diagnostic context (dict + text), and returns a JSON-safe result.
    Logging is attempted only when request+user are available.
    """
    try:
        start_time = datetime.datetime.now()

        seg_model = get_segmentation_model()
        plus_model = get_classification_model()
        stage_model = get_stage_model()
        zone_model = get_zone_model()

        # --- 1) Segmentation mask for Plus classifier input ---
        image_file.seek(0)
        mask = predict_mask(image_file, seg_model, device)
        _, mask_buffer = cv2.imencode('.jpg', mask)
        mask_bytes = mask_buffer.tobytes()

        # --- 2) Plus / No-Plus ---
        class_name, class_prob = get_prediction(mask_bytes, plus_model)

        # --- 3) Stage ---
        image_file.seek(0)
        image_bytes = image_file.read()
        stage_tensor = transform_image(image_bytes).to(device)
        with torch.inference_mode():
            stage_output = stage_model(stage_tensor)
            stage_probs = torch.softmax(stage_output, dim=1)
            stage_label = torch.argmax(stage_probs, dim=1).item()
        stage_names = ['Normal', 'Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5']
        stage_result = {
            "stage_name": stage_names[stage_label],
            "stage_prob": f"{stage_probs[0, stage_label].item():.3f}"
        }

        # --- 4) Zone (use first 2 logits for Z1/Z2; fallback to Z3 if low confidence) ---
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

        zone_result = {
            "zone_name": zone_label,
            "zone_prob": f"{zone_conf:.3f}"
        }

        # --- 5) Final decision (hard rules) ---
        final_decision = compute_final_decision(
            zone_label=zone_label,
            stage_label=stage_result["stage_name"],
            plus_label=class_name
        )

        # --- 6) Overlay vessels for visualization ---
        image_file.seek(0)
        segmented_image = vessels(image_file, seg_model, device)
        _, seg_buffer = cv2.imencode('.jpg', segmented_image)
        encoded_string = base64.b64encode(seg_buffer.tobytes()).decode('utf-8')
        image_data = f'data:image/jpeg;base64,{encoded_string}'

        end_time = datetime.datetime.now()
        execution_time = f'{round((end_time - start_time).total_seconds() * 1000)} ms'
        file_name = getattr(image_file, "name", "uploaded_image.jpg")

        # --- 7) Unified diagnostic context (dict + human text) ---
        diag_ctx = {
            "predictions": {
                "class_name": class_name,
                "class_prob": f"{class_prob:.3f}"
            },
            "stage_prediction": stage_result,
            "zone_prediction": zone_result,
            "final_decision": final_decision
        }
        diag_ctx_text = (
            f"Plus disease: {diag_ctx['predictions']['class_name']} "
            f"(p={diag_ctx['predictions']['class_prob']}). "
            f"Stage: {diag_ctx['stage_prediction']['stage_name']} "
            f"(p={diag_ctx['stage_prediction']['stage_prob']}). "
            f"Zone: {diag_ctx['zone_prediction']['zone_name']} "
            f"(p={diag_ctx['zone_prediction']['zone_prob']}). "
            f"Final decision: {diag_ctx['final_decision']}."
        )

        result = {
            "image_data": image_data,
            "inference_time": execution_time,
            "file_name": file_name,
            "predictions": {
                "class_name": class_name,
                "class_prob": f"{class_prob:.3f}"
            },
            "stage_prediction": stage_result,
            "zone_prediction": zone_result,
            "final_decision": final_decision,
            "diagnostic_context": diag_ctx,
            "diagnostic_context_text": diag_ctx_text,
        }

        # --- 8) Optional logging (only if request & user available) ---
        if request is not None and hasattr(request, "user") and getattr(request.user, "is_authenticated", False):
            from .models import PredictionLog
            image_file.seek(0)

            log = PredictionLog.objects.create(
                user=request.user,
                file_name=file_name,
                predicted_class=class_name,
                probability=float(class_prob),
                stage_class=stage_result["stage_name"],
                stage_probability=float(stage_result["stage_prob"]),
                zone_class=zone_label,
                zone_probability=float(zone_result["zone_prob"]),
                final_decision=final_decision,
                execution_time=round((end_time - start_time).total_seconds() * 1000)
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

            # Include URLs back in result if available
            if getattr(log, "image_url", None):
                result["image_url"] = log.image_url
            if getattr(log, "segmented_image_url", None):
                result["segmented_image_url"] = log.segmented_image_url
        llm_summary = (
            f"Plus: {class_name} (p={class_prob:.3f}); "
            f"Stage: {stage_result['stage_name']} (p={stage_result['stage_prob']}); "
            f"Zone: {zone_result['zone_name']} (p={zone_result['zone_prob']}); "
            f"Final Decision: {final_decision}"
        )
        result["llm_diagnostic_text"] = llm_summary

        return result

    except Exception as e:
        print(f"Error in get_result: {e}")
        raise e
