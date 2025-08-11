import joblib
import json
import pandas as pd
MODEL_PATH_HYPERTENSION = 'model/best_rf_hypertension_model.joblib'


def predict_hypertension_risk(
        male, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
        totChol, sysBP, diaBP, BMI, heartRate, glucose, city=None, region=None, insurance=None,
        # <-- ADD 'province' HERE
        model_path=MODEL_PATH_HYPERTENSION
):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        # This now returns a JSON STRING, which is valid for the API
        error_payload = {"error": f"Model file not found at path: {model_path}"}
        return json.dumps(error_payload)
    input_data = pd.DataFrame([{
        "male": male, "age": age, "currentSmoker": currentSmoker, "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds, "diabetes": diabetes, "totChol": totChol, "sysBP": sysBP,
        "diaBP": diaBP, "BMI": BMI, "heartRate": heartRate, "glucose": glucose
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_class_index = list(model.classes_).index(prediction)
    predicted_probability = probabilities[predicted_class_index]
    percentage = round(predicted_probability * 100, 2)

    if prediction == 1:
        return f"⚠️ Based on the model, the patient has a {percentage}% probability of **having hypertension**."
    else:
        return f"✅ Based on the model, the patient has a {percentage}% probability of **not having hypertension**."

