import io
import base64
import datetime
import tempfile
from PIL import Image
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import numpy as np
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b6

import torchvision
from django.core.files.base import ContentFile
import uuid
from segmentation_models_pytorch import UnetPlusPlus
# allow-list target for PyTorch safe loader
try:
    from torchvision.models.efficientnet import EfficientNet
except Exception:
    EfficientNet = None  # we'll handle fallback below


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global models (lazy-loaded)
model = None
model_efficient_b4 = None

# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
# ---- Robust checkpoint loader (handles PyTorch 2.6, state_dict or full module) ----
# ---- Robust checkpoint loader (PyTorch 2.6-safe, state_dict OR full module) ----
def load_checkpoint_forgiving(model: nn.Module, path: str, device, strict: bool = False):
    # Try safe load first, allow-list EfficientNet if available
    try:
        if EfficientNet is not None:
            # Prefer context manager to limit scope of allow-listing
            with torch.serialization.safe_globals([EfficientNet]):
                obj = torch.load(path, map_location=device, weights_only=True)
        else:
            # Fallback: try to infer EfficientNet class from a dummy instance
            try:
                dummy = efficientnet_b4(weights=None)
                with torch.serialization.safe_globals([type(dummy)]):
                    obj = torch.load(path, map_location=device, weights_only=True)
            except Exception:
                obj = torch.load(path, map_location=device, weights_only=True)
    except Exception as e_safe:
        # Fall back to full unpickle (ONLY if you trust the file/source)
        try:
            obj = torch.load(path, map_location=device, weights_only=False)
        except Exception as e_unsafe:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}'.\n"
                f"Safe loader error: {e_safe}\n"
                f"Unsafe loader error: {e_unsafe}"
            )

    # If the checkpoint is an entire module, just use it
    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model

    # Otherwise, resolve a state_dict from common layouts
    if isinstance(obj, dict):
        # common keys: 'state_dict', 'model_state_dict', 'module', 'model'
        for key in ("state_dict", "model_state_dict", "module", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
        else:
            state = obj  # assume it's already a state_dict

        # Strip 'module.' prefixes if saved with DataParallel/DistributedDataParallel
        if any(k.startswith("module.") for k in state.keys()):
            from collections import OrderedDict
            state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())

        # Load with your chosen strictness
        model.load_state_dict(state, strict=strict)
        model = model.to(device).eval()
        return model

    # Unknown format
    raise RuntimeError(f"Unexpected checkpoint object type: {type(obj)} for '{path}'")

    # obj is a dict -> pull state_dict if present
    state = obj.get("state_dict", obj)
    model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    return model

class_names = ["No Plus", "Plus"]

# --------------------------
# Lazy-loading model getters
# --------------------------
def get_segmentation_model():
    global model
    if model is None:
        mask_model = UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)

        weight_path = "model/best_weight_Unet++_maskresize_29"
        mask_model = load_checkpoint_forgiving(mask_model, weight_path, device, strict=False)
        model = mask_model
    return model

def get_classification_model():
    global model_efficient_b4
    if model_efficient_b4 is None:
        model_efficient_b4 = efficientnet_b4(weights=None).to(device)
        model_efficient_b4.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1792, 2, bias=True)
        ).to(device)

        best_model_path = "model/model_efficentnet_b4_plus.pth"
        model_efficient_b4 = load_checkpoint_forgiving(model_efficient_b4, best_model_path, device, strict=False)
    return model_efficient_b4


# --------------------------
# UNet definition
# --------------------------
# class conv_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_c)
#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_c)
#         self.relu = nn.ReLU()
#
#     def forward(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         return x
#
#
# class encoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = conv_block(in_c, out_c)
#         self.pool = nn.MaxPool2d((2, 2))
#
#     def forward(self, inputs):
#         x = self.conv(inputs)
#         p = self.pool(x)
#         return x, p
#
#
# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
#         self.conv = conv_block(out_c + out_c, out_c)
#
#     def forward(self, inputs, skip):
#         x = self.up(inputs)
#         x = torch.cat([x, skip], axis=1)
#         x = self.conv(x)
#         return x
#
#
# class build_unet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.e1 = encoder_block(3, 64)
#         self.e2 = encoder_block(64, 128)
#         self.e3 = encoder_block(128, 256)
#         self.e4 = encoder_block(256, 512)
#         self.b = conv_block(512, 1024)
#         self.d1 = decoder_block(1024, 512)
#         self.d2 = decoder_block(512, 256)
#         self.d3 = decoder_block(256, 128)
#         self.d4 = decoder_block(128, 64)
#         self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
#
#     def forward(self, inputs):
#         s1, p1 = self.e1(inputs)
#         s2, p2 = self.e2(p1)
#         s3, p3 = self.e3(p2)
#         s4, p4 = self.e4(p3)
#         b = self.b(p4)
#         d1 = self.d1(b, s4)
#         d2 = self.d2(d1, s3)
#         d3 = self.d3(d2, s2)
#         d4 = self.d4(d3, s1)
#         outputs = self.outputs(d4)
#         return outputs
#
#
# --------------------------
# Image processing functions
# --------------------------
def predict_mask(image_file, model, device, size=(512, 512)):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
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
    purple = np.array([128, 0, 128], dtype=np.uint8)  # بنفش ملایم
    alpha = 0.7  # شفافیت

    result_image[vessel_mask == 255] = (
            alpha * purple + (1 - alpha) * result_image[vessel_mask == 255]
    ).astype(np.uint8)

    return result_image


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return simple_transform(image).unsqueeze(dim=0)


def get_prediction(image_bytes, model_efficient_b4):
    tensor = transform_image(image_bytes).to(device)

    with torch.inference_mode():
        outputs = model_efficient_b4(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    return class_names[pred_label], probs[0, pred_label].item()


# --------------------------
# Main inference entry point
# --------------------------
def get_result(image_file, is_api=False, request=None):
    try:
        start_time = datetime.datetime.now()

        segmentation_model = get_segmentation_model()
        classification_model = get_classification_model()

        image_file.seek(0)
        mask = predict_mask(image_file, segmentation_model, device)

        _, buffer = cv2.imencode('.jpg', mask)
        mask_bytes = buffer.tobytes()
        class_name, class_prob = get_prediction(mask_bytes, classification_model)
        stage_model = get_stage_model()
        image_file.seek(0)
        image_bytes = image_file.read()

        stage_tensor = transform_image(image_bytes).to(device)
        with torch.inference_mode():
            stage_output = stage_model(stage_tensor)
            stage_probs = torch.softmax(stage_output, dim=1)
            stage_label = torch.argmax(stage_probs, dim=1).item()

        stage_names = ['Normal','Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5']
        stage_result = {
            "stage_name": stage_names[stage_label],
            "stage_prob": f"{stage_probs[0, stage_label].item():.3f}"
        }
        zone_model = get_zone_model()
        zone_tensor = transform_image(image_bytes).to(device)

        with torch.inference_mode():
            zone_logits_full = zone_model(zone_tensor)          # shape [1, 7]
            zone_logits = zone_logits_full[:, :2]               # use first two logits only
            zone_probs = torch.softmax(zone_logits, dim=1)      # probs for Zone1/Zone2
            max_prob, pred_12 = torch.max(zone_probs, dim=1)

        if max_prob.item() < 0.5:
            zone_label = "Zone 3"
            zone_conf  = 1.0 - max_prob.item()                  # optional: a proxy
        else:
            zone_label = "Zone 1" if pred_12.item() == 0 else "Zone 2"
            zone_conf  = max_prob.item()
        zone_result = {
            "zone_name": zone_label,
            "zone_prob": f"{zone_conf:.3f}"
        }
        # ---- Final Decision (exact Colab logic) ----
        plus_label = class_name  # class_name is "No Plus" or "Plus"
        stage_label_str = stage_result["stage_name"]
        zone_label_str  = zone_result["zone_name"]

        final_decision = compute_final_decision(
            zone_label=zone_label_str,
            stage_label=stage_label_str,
            plus_label=plus_label
        )

        image_file.seek(0)
        segmented_image = vessels(image_file, segmentation_model, device)
        _, buffer = cv2.imencode('.jpg', segmented_image)
        encoded_string = base64.b64encode(buffer.tobytes())
        bs64 = encoded_string.decode('utf-8')
        image_data = f'data:image/jpeg;base64,{bs64}'

        end_time = datetime.datetime.now()
        execution_time = f'{round((end_time - start_time).total_seconds() * 1000)} ms'

        file_name = image_file.name

        result = {
            "image_data": image_data,
            "inference_time": execution_time,
            "predictions": {
                "class_name": class_name,
                "class_prob": f"{class_prob:.3f}"
            },
            "file_name": file_name
        }

        # Save log
        from .models import PredictionLog
        image_file.seek(0)

        log = PredictionLog.objects.create(
            user=request.user,
            file_name=file_name,
            predicted_class=class_name,
            probability=class_prob,
            stage_class=stage_names[stage_label],
            stage_probability=stage_probs[0, stage_label].item(),
            zone_class=zone_label,
            zone_probability=zone_conf,
            final_decision=final_decision,                    # NEW
            execution_time=round((end_time - start_time).total_seconds() * 1000)
        )


        seg_image_name = f"segmented_{uuid.uuid4().hex}.jpg"
        seg_image_content = ContentFile(buffer.tobytes(), name=seg_image_name)

        log.segmented_image.save(seg_image_name, seg_image_content)
        log.segmented_image_url = request.build_absolute_uri(log.segmented_image.url)
        log.save(update_fields=["segmented_image_url"])
        log.image = image_file
        log.save()
        log.image_url = request.build_absolute_uri(log.image.url)
        log.save(update_fields=["image_url"])
        result["stage_prediction"] = stage_result

        result["image_url"] = log.image_url
        result["stage_prediction"] = stage_result
        result["zone_prediction"]  = zone_result
        result["stage_prediction"] = stage_result
        result["zone_prediction"]  = zone_result
        result["final_decision"]   = final_decision          # NEW
        # NEW

        return result

    except Exception as e:
        print(f"Error in get_result: {e}")
        raise e
# --------------------------
# Lazy-load stage classification model
# --------------------------
model_stage = None
def get_stage_model():
    global model_stage
    if model_stage is None:
        model_stage = efficientnet_b6(weights=None).to(device)
        model_stage.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(2304, 7, bias=True)
        ).to(device)

        best_model_path = "model/best_model (1).pth"
        model_stage = load_checkpoint_forgiving(model_stage, best_model_path, device, strict=False)
    return model_stage
model_zone = None
def get_zone_model():
    global model_zone
    if model_zone is None:
        zone_model = efficientnet_b4(weights=None).to(device)
        zone_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1792, 7, bias=True)
        ).to(device)

        zone_weights_path = "model/model_Zone_augment_Farabi_2"
        model_zone = load_checkpoint_forgiving(zone_model, zone_weights_path, device, strict=False)
    return model_zone

def compute_final_decision(zone_label: str, stage_label: str, plus_label: str) -> str:
    # Hard rules from your Colab
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

    # Fallback (should not happen if labels are valid)
    return "Follow-up"
