"""Hypertension risk predictor (scikit-learn / joblib).

Ported from the original ``test_analysis/disease_models.py``. The only change is
that the model path defaults from :mod:`config` instead of a hard-coded string;
the feature engineering and output strings are unchanged so predictions match.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from .config import get_settings

# The 12 numeric features the model was trained on, in order.
_FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]


def predict_hypertension_risk(
    male, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
    totChol, sysBP, diaBP, BMI, heartRate, glucose,
    city=None, region=None, insurance=None,  # accepted for tool-call parity; unused by the model
    model_path: str | None = None,
) -> str:
    """Return a human-readable hypertension-risk statement.

    Args mirror the model's training columns. The ``city``/``region``/``insurance``
    args are accepted (the LLM tool schema passes them) but are not model inputs.

    Returns a Persian/English risk sentence, or a JSON error string if the model
    file is missing — both are valid tool outputs for the LLM pipeline.
    """
    model_path = model_path or get_settings().hypertension_model_path
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return json.dumps({"error": f"Model file not found at path: {model_path}"})

    input_data = pd.DataFrame([{
        "male": male, "age": age, "currentSmoker": currentSmoker, "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds, "diabetes": diabetes, "totChol": totChol, "sysBP": sysBP,
        "diaBP": diaBP, "BMI": BMI, "heartRate": heartRate, "glucose": glucose,
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_class_index = list(model.classes_).index(prediction)
    percentage = round(probabilities[predicted_class_index] * 100, 2)

    if prediction == 1:
        return f"⚠️ Based on the model, the patient has a {percentage}% probability of **having hypertension**."
    return f"✅ Based on the model, the patient has a {percentage}% probability of **not having hypertension**."
