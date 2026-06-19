# How to Run Single-ROP Service

Automated detection & classification of **Retinopathy of Prematurity (ROP)** from
retinal fundus images. Runs a four-model pipeline (vessel segmentation →
Plus-disease → stage → zone) and returns a clinical recommendation plus guidance.

## Prerequisites
- Python 3.10+
- ~2 GB free RAM (CPU inference) — a CUDA GPU is used automatically if present.
- The **four model checkpoints** are bundled in this service's `model/` directory
  and are loaded automatically — no configuration needed:
  - `best_weight_Unet++_maskresize_29`  (segmentation)
  - `model_efficentnet_b4_plus.pth`      (Plus / No-Plus)
  - `best_model (1).pth`                 (stage)
  - `model_Zone_augment_Farabi_2`        (zone)

  Set `MODEL_DIR` only if you want to load them from a different location.

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
All configuration is via environment variables — nothing is hard-coded.

| Variable             | Default  | Description                                              |
|----------------------|----------|----------------------------------------------------------|
| `MODEL_DIR`          | bundled `model/` | Directory containing the four checkpoint files (above). |
| `ROP_DEVICE`         | *(auto)* | Force a torch device, e.g. `cpu` or `cuda:0`.            |
| `ROP_PORT`           | `8001`   | Port for the convenience launcher (`python routes.py`).  |
| `ROP_ZONE_THRESHOLD` | `0.5`    | Confidence below which the zone head falls back to Zone 3.|

Example (Linux/Mac), pointing at the original repo's model folder:
```bash
export MODEL_DIR=/home/amir/Desktop/rop/model
```

## Run
Run from **inside the `ghofran/` directory** so the package import resolves
(`single_rop` is a package, so we expose it as `single_rop.routes:app`):

```bash
cd ..                      # into ghofran/
uvicorn single_rop.routes:app --host 0.0.0.0 --port 8001
```

Alternatively, the module is self-launching:
```bash
cd single_rop
python routes.py           # honours ROP_PORT (default 8001)
```

Once running, open the interactive docs at: <http://localhost:8001/docs>

## Test

**1. Liveness check**
```bash
curl http://localhost:8001/health
# -> {"status":"ok","service":"single_rop"}
```

**2. Single image prediction**
```bash
curl -X POST http://localhost:8001/predict \
  -F "files=@/path/to/fundus.jpg"
```

**3. Multiple images (batch + aggregation)**
```bash
curl -X POST http://localhost:8001/predict \
  -F "files=@/path/to/eye1.jpg" \
  -F "files=@/path/to/eye2.jpg"
```

Expected response shape:
```json
{
  "aggregated": {
    "predictions": {"class_name": "Plus", "class_prob": "0.985"},
    "stage_prediction": {"stage_name": "Stage 3", "stage_prob": "0.942"},
    "zone_prediction": {"zone_name": "Zone 2", "zone_prob": "0.880"},
    "final_decision": "Treatment",
    "guidance": {"title": "...", "text": "..."},
    "worst_index": 0
  },
  "per_image": [ { "...": "one result object per uploaded image" } ]
}
```

> Tip: the `image_data` / `original_image_data` fields are base64 data-URIs you
> can drop straight into an `<img src="...">` tag to view the vessel overlay.
```
