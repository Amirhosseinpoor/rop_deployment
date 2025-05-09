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
import torchvision
from django.core.files.base import ContentFile
import uuid
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

class_names = ["Normal", "Plus"]

# --------------------------
# Lazy-loading model getters
# --------------------------
def get_segmentation_model():
    global model
    if model is None:
        model = build_unet()
        model = model.to(device)
        checkpoint_path = 'model/checkpoint.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    return model


def get_classification_model():
    global model_efficient_b4
    if model_efficient_b4 is None:
        model_efficient_b4 = efficientnet_b4(weights=None).to(device)

        model_efficient_b4.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1792, out_features=2, bias=True)
        ).to(device)
        best_model_path = "model/model_efficentnet_b4_plus.pth"
        model_efficient_b4.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        model_efficient_b4.eval()
    return model_efficient_b4


# --------------------------
# UNet definition
# --------------------------
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.b = conv_block(512, 1024)
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs


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

        stage_names = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3']
        stage_result = {
            "stage_name": stage_names[stage_label],
            "stage_prob": f"{stage_probs[0, stage_label].item():.3f}"
        }

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
        model_stage = torchvision.models.efficientnet_b4(weights=None).to(device)
        model_stage.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1792, out_features=4, bias=True)
        ).to(device)
        model_stage.load_state_dict(torch.load('model/model_eff_b4_3_stage.pth', map_location=device))
        model_stage.eval()
    return model_stage
