import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

model = joblib.load("models/model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
preprocessing = joblib.load("models/preprocessing.pkl")

FEATURE_ORDER = preprocessing["feature_order"]
CATEGORICAL_COLS = set(preprocessing["categorical_cols"])
NUMERIC_FILL_VALUES = preprocessing["numeric_fill_values"]


# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_input(data: dict) -> pd.DataFrame:
    processed = []

    for feature in FEATURE_ORDER:
        value = data.get(feature)

        if feature in CATEGORICAL_COLS:
            if value is None:
                value = "Unknown"
            value = str(value)

            encoder = label_encoders[feature]

            if value not in encoder.classes_:
                allowed_values = ", ".join(map(str, encoder.classes_))
                raise ValueError(
                    f"Invalid value for '{feature}': '{value}'. Allowed values: {allowed_values}"
                )

            processed.append(int(encoder.transform([value])[0]))
            continue

        if value is None:
            value = NUMERIC_FILL_VALUES[feature]

        processed.append(float(value))

    return pd.DataFrame([processed], columns=FEATURE_ORDER, dtype=float)


# -----------------------------
# Serve UI (NEW)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("frontend/index.html", "r") as f:
        return f.read()


# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
def predict(data: dict):
    try:
        features = preprocess_input(data)
        prediction = model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "risk": "High Risk" if prediction == 1 else "Low Risk"
        }

    except Exception as e:
        return {"error": str(e)}