# How to Run Double-ROP Service

Binocular keratoconus / corneal classification. Takes a **pair** of eye images
(left + right) and returns a per-eye class, a combined binocular "Z" class, and
base64 previews of the inputs.

## Prerequisites
- Python 3.10+
- ~2 GB free RAM (CPU inference) — a CUDA GPU is used automatically if present.
- The trained **EyeNet checkpoint** `best_model.pth` (a `resnet50`-based
  state_dict) is bundled in this service's `model/` directory and loaded
  automatically. Override with `KC_MODEL_PATH`/`MODEL_DIR` if needed.

> First run also downloads the torchvision ImageNet weights for ResNet-50
> (used to initialise the two encoders). Allow internet access for that, or
> pre-populate the torch hub cache.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration (environment variables)

| Variable       | Default                   | Description                                   |
|----------------|---------------------------|-----------------------------------------------|
| `KC_MODEL_PATH`| `${MODEL_DIR}/best_model.pth` | Full path to the EyeNet checkpoint.       |
| `MODEL_DIR`    | `model`                   | Used to build the default `KC_MODEL_PATH`.    |
| `KC_DEVICE`    | *(auto)*                  | Force a device, e.g. `cpu` or `cuda:0`.       |
| `KC_PORT`      | `8002`                    | Port for the convenience launcher.            |

Example pointing at the original repo's checkpoint:
```bash
export KC_MODEL_PATH=/home/amir/Desktop/rop/model/best_model.pth
```

## Run
Run from inside the `ghofran/` directory so the package import resolves:
```bash
cd ..                       # into ghofran/
uvicorn double_rop.routes:app --host 0.0.0.0 --port 8002
```

Or self-launch the module (honours `KC_PORT`):
```bash
cd double_rop
python routes.py
```

Interactive docs: <http://localhost:8002/docs>

## Test

**1. Liveness check**
```bash
curl http://localhost:8002/health
# -> {"status":"ok","service":"double_rop"}
```

**2. Prediction (both files required)**
```bash
curl -X POST http://localhost:8002/predict \
  -F "left_file=@/path/to/left_eye.png" \
  -F "right_file=@/path/to/right_eye.png"
```

Expected response:
```json
{
  "left_eye":  {"label": "Normal", "probability": "0.9912"},
  "right_eye": {"label": "eKCN",   "probability": "0.8740"},
  "z_class":   {"label": "NSfRS",  "probability": "0.9531"},
  "image_data": {"left": "data:image/png;base64,...", "right": "data:image/png;base64,..."},
  "inference_time": 0.42
}
```

Class meanings:
- Per-eye classes (`left_eye`/`right_eye`): `Normal`, `ATN`, `NEIr`, `EIr`, `eKCN`.
- Binocular Z class (`z_class`): `SfRS`, `NSfRS`.
```
