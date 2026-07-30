"""
Standalone loader / inference script for the EYES-DEFY-ANEMIA Phase 4
classification checkpoints (ResNet18, MobileNetV3-Small, EfficientNet-B0).

This file has NO dependency on the rest of the project repo -- it only
needs the .pth checkpoint file(s) sitting next to it (or pointed to via
--checkpoint). Install requirements:

    pip install torch torchvision albumentations pillow numpy

--------------------------------------------------------------------------
IMPORTANT -- expected input image format
--------------------------------------------------------------------------
These models were NOT trained on raw eye photos. They were trained on
pre-cropped, pre-flattened 256x256 tissue crops (either the "palpebral" or
"forniceal_palpebral" conjunctiva region, already isolated against a clean
black background -- see prepare_dataset.py in the original repo). Feeding
a raw, uncropped eye photo through these models is out-of-distribution and
will not give meaningful results. This script's preprocessing (resize +
ImageNet normalize) matches training exactly, but it does NOT crop/isolate
the conjunctiva for you -- the input image must already be that crop.

--------------------------------------------------------------------------
Checkpoint naming -> (architecture, tissue_type)
--------------------------------------------------------------------------
best_resnet18_palpebral.pth                    -> resnet18 / palpebral
best_resnet18_forniceal_palpebral.pth           -> resnet18 / forniceal_palpebral
best_mobilenet_v3_small_palpebral.pth           -> mobilenet_v3_small / palpebral
best_mobilenet_v3_small_forniceal_palpebral.pth -> mobilenet_v3_small / forniceal_palpebral
best_efficientnet_b0_palpebral.pth              -> efficientnet_b0 / palpebral
best_efficientnet_b0_forniceal_palpebral.pth    -> efficientnet_b0 / forniceal_palpebral

The tissue_type only matters for knowing what kind of crop to feed in --
it does not change the model code below.

Best overall (highest F1/Acc/AUC, all 0.903): efficientnet_b0 / forniceal_palpebral
Best confound-handling (smallest India/Italy AUC gap): mobilenet_v3_small / forniceal_palpebral

--------------------------------------------------------------------------
Output convention
--------------------------------------------------------------------------
Each model has a single output logit (no Sigmoid inside the model). Label
convention: 1 = anemic, 0 = non-anemic (WHO Hgb threshold: male < 13.0
g/dL, female < 12.0 g/dL). Apply torch.sigmoid() to the logit to get
P(anemic), then threshold at 0.5 for a hard prediction.
"""

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import models

# --------------------------------------------------------------------------
# 1. Model architectures -- exact match to the original training setup
# --------------------------------------------------------------------------
# `weights=None` here is deliberate, not a simplification: at training time
# each backbone started from ImageNet-pretrained weights with everything
# frozen except the replaced head, but the saved checkpoint's state_dict
# contains ALL parameters and buffers (frozen backbone included) at their
# final trained values. Loading that state_dict overwrites every weight
# regardless of how the architecture was initialized first, so building
# with weights=None avoids an unnecessary internet download and makes this
# script work fully offline.


def build_resnet18() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def build_mobilenet_v3_small() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 1)
    return model


def build_efficientnet_b0() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


ARCHITECTURE_REGISTRY = {
    "resnet18": build_resnet18,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "efficientnet_b0": build_efficientnet_b0,
}

# --------------------------------------------------------------------------
# 2. Preprocessing -- exact match to get_eval_transforms() in dataset.py
#    (no random flip/rotation -- that's train-only augmentation)
# --------------------------------------------------------------------------
IMAGE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

eval_transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL Image -> normalized [1, 3, 256, 256] tensor, batch dim included."""
    image_np = np.array(image.convert("RGB"))
    tensor = eval_transform(image=image_np)["image"]  # [3, 256, 256]
    return tensor.unsqueeze(0)  # [1, 3, 256, 256]


# --------------------------------------------------------------------------
# 3. Loading a checkpoint
# --------------------------------------------------------------------------
def load_model(checkpoint_path: str, arch_name: str, device: torch.device) -> nn.Module:
    if arch_name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"arch_name must be one of {list(ARCHITECTURE_REGISTRY)}, got {arch_name!r}")

    model = ARCHITECTURE_REGISTRY[arch_name]()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)  # strict=True by default -- shapes must match exactly
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------
# 4. Inference
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model: nn.Module, image: Image.Image, device: torch.device) -> dict:
    input_tensor = preprocess_image(image).to(device)
    logit = model(input_tensor).squeeze()  # scalar
    prob_anemic = torch.sigmoid(logit).item()
    return {
        "prob_anemic": prob_anemic,
        "predicted_label": "Anemic" if prob_anemic > 0.5 else "Non-anemic",
    }


@torch.no_grad()
def run_dummy_inference(model: nn.Module, device: torch.device) -> dict:
    """Sanity check that a checkpoint loads and produces a valid output,
    without needing a real image on hand -- a random tensor of the right
    shape is enough to prove the architecture/weights line up."""
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    logit = model(dummy_input).squeeze()
    prob = torch.sigmoid(logit).item()
    print(f"[dummy inference] output logit={logit.item():.4f}, sigmoid(logit)={prob:.4f}")
    return {"prob_anemic": prob}


# --------------------------------------------------------------------------
# 5. Example usage
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Load a Phase 4 classification checkpoint and run inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to a best_*.pth file")
    parser.add_argument(
        "--arch",
        required=True,
        choices=list(ARCHITECTURE_REGISTRY),
        help="Architecture matching the checkpoint (see naming table in this file's docstring)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a pre-cropped palpebral/forniceal_palpebral tissue image. "
        "If omitted, runs a dummy random-tensor inference instead (just to prove the checkpoint loads).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, args.arch, device)
    print(f"Loaded {args.arch} from {args.checkpoint}")

    if args.image is None:
        print("No --image given, running a dummy inference to confirm the model loaded correctly...")
        run_dummy_inference(model, device)
    else:
        image = Image.open(args.image)
        result = predict(model, image, device)
        print(f"P(anemic) = {result['prob_anemic']:.4f}")
        print(f"Predicted label: {result['predicted_label']}")


if __name__ == "__main__":
    main()
