# Double ROP: Pairwise KC & Multi-Eye Analysis

The `double_rop` application extends the platform's vision capabilities to **Keratoconus (KC) eye disease diagnostics** and coordinated analysis of binocular ROP images.

---

### 1. Models (`models.py`)
Diagnostic records are persisted in the **`PredictionResult`** table.

- **Schema Highlights**:
    - **Eye-Specific Labels**: `left_label`, `right_label` (KC classification) with corresponding `_probability` fields.
    - **Z-Class**: `z_class_label` (Stability assessment: SfRS/NSfRS).
    - **Image Storage**: `left_image`, `right_image` fields for topography or fundus pairs.
    - **Feedback Loop**: Includes `corrected_left_label`, `corrected_right_label`, and `corrected_z_label` to capture clinician corrections for misclassified KC cases.
    - **Status**: `classification_status` tracks audit state (1 = Correct, -1 = Incorrect).

---

### 2. Core Logic & AI Pipeline (`utils.py`)
The core of this module is **EyeNet**, a sophisticated Siamese-style architecture designed for ocular symmetry analysis.

#### **Pipeline Trace:**
1.  **Preprocessing**: Images are resized to `(224, 224)` and normalized.
2.  **Backbone (ResNet50 + CBAM + Transformers)**:
    - Parallel branches process left and right eye images.
    - **CBAM (Convolutional Block Attention Module)**: Enhances spatial and channel features.
    - **Bottleneck Transformer**: Captures long-range dependencies in ocular textures.
3.  **Inference Logic**:
    - **Left/Right classification**: Predicts one of 5 classes (`Normal`, `ATN`, `NEIr`, `EIr`, `eKCN`).
    - **Z-Classification**: Concatenates left and right features into a shared `z_fc` layer to predict stability (Symmetry vs Asymmetry).

#### **I/O Example:**
- **Input**: Two corneal topography images (Left & Right).
- **Output Dictionary**:
  ```json
  {
    "left_eye": {"label": "eKCN", "probability": "0.9124"},
    "right_eye": {"label": "Normal", "probability": "0.9855"},
    "z_class": {"label": "NSfRS", "probability": "0.9412"}
  }
  ```

---

### 3. Views & Handlers (`views.py`)
- **`home(request)`**:
    - **Feedback Logic**: Handles `POST` requests with `feedback_mode`. It retrieves the latest `PredictionResult` for the user and updates it with "corrected" labels provided by the clinician.
    - **Inference Mode**: Processes `left_file` and `right_file`, calls `get_prediction`, and saves a new `PredictionResult` record.
- **`predict(request)`**:
    - An anonymous `@csrf_exempt` endpoint providing raw JSON results for the EyeNet pipeline.

---

### 4. Routing (`urls.py`)
| Endpoint | View | Logic |
| :--- | :--- | :--- |
| `/` | `views.home` | Dashboard for KC diagnostics and feedback. |
| `/predict/` | `views.predict` | Stateless JSON API for KC inference. |

---
*Note: This module handles its corresponding frontend views via template rendering in the `templates/` directory.*
