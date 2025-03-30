import io
import base64
import datetime
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the normalization transform (aligned with ImageNet)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # [R, G, B]
    std=[0.229, 0.224, 0.225],  # [R, G, B]
)

# Compose transformations
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size for EfficientNet-B3
    transforms.ToTensor(),          # Convert to tensor with [0, 1] range
    normalize                       # Normalize using ImageNet values
])

# Load the EfficientNet-B3 model with custom classifier
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
weights_eff_b4 = EfficientNet_B4_Weights.IMAGENET1K_V1  # Use IMAGENET1K_V1 directly
model_efficient_b4 = efficientnet_b4(weights=weights_eff_b4).to(device)

#weights_eff_b4 = torchvision.models.EfficientNet_B4_Weights.DEFAULT
#model_efficient_b4 = torchvision.models.efficientnet_b4(weights=weights_eff_b4).to(device)
model_efficient_b4.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1792,
                    out_features=2,
                    bias=True)).to(device)

# Load the trained model weights
best_model_path = "model/model_efficentnet_b4_plus.pth"  # Update with the actual path
model_efficient_b4.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
model_efficient_b4.eval()

# Define class names (hardcoded)
class_names = ["Plus", "Normal_comp"]


def transform_image(image_bytes):
    """Apply image transformations using simple_transform."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Ensure RGB
    return simple_transform(image).unsqueeze(dim=0)


def get_prediction(image_bytes):
    """Predict the class label and probability for an input image."""
    tensor = transform_image(image_bytes).to(device)

    with torch.inference_mode():
        outputs = model_efficient_b4(tensor)
        probs = torch.softmax(outputs, dim=1)  # Get probabilities
        pred_label = torch.argmax(probs, dim=1).item()  # Get the predicted class index

    return class_names[pred_label], probs[0, pred_label].item()  # Return class and probability


def get_result(image_file, is_api=False):
    """Format the result, including inference time and (optional) base64 image encoding."""
    try:
        start_time = datetime.datetime.now()
        image_bytes = image_file.file.read()



        class_name, class_prob = get_prediction(image_bytes)
        end_time = datetime.datetime.now()

        # Compute inference time
        time_diff = (end_time - start_time)
        execution_time = f'{round(time_diff.total_seconds() * 1000)} ms'

        # Encode the image for the frontend if needed
        encoded_string = base64.b64encode(image_bytes)
        bs64 = encoded_string.decode('utf-8')
        image_data = f'data:image/jpeg;base64,{bs64}'

        # Extract the file name
        file_name = image_file.name  # Updated this line to use `name` instead of `filename`

        # Prepare the result
        result = {
            "inference_time": execution_time,
            "predictions": {
                "class_name": class_name,
                "class_prob": f"{class_prob:.3f}"  # Format the probability
            },
            "file_name": file_name  # Updated this line to use `file_name`
        }
        from .models import PredictionLog  # بالا اضافه کن

        PredictionLog.objects.create(
            file_name=file_name,
            predicted_class=class_name,
            probability=class_prob,
            execution_time=round(time_diff.total_seconds() * 1000)
        )

        if not is_api:
            result["image_data"] = image_data


        return result

    except Exception as e:
        print(f"Error in get_result: {e}")
        raise e