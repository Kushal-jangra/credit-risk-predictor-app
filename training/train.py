import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = os.path.join("data", "credit_risk_dataset.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded")
print(df.head())


# -----------------------------
# 2. Preprocessing
# -----------------------------

# Handle missing values by dtype to avoid mixing strings into numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna("Unknown", inplace=True)

# Encode categorical columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# -----------------------------
# 3. Feature & Target Split
# -----------------------------
y = df["loan_status"]
X = df.drop(columns=["loan_status"])


# -----------------------------
# 4. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 5. Model Training
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")


# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")


# -----------------------------
# 7. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(
    {
        "feature_order": X.columns.tolist(),
        "categorical_cols": list(categorical_cols),
        "numeric_fill_values": {col: float(df[col].median()) for col in numeric_cols if col != "loan_status"},
    },
    "models/preprocessing.pkl",
)

print("Model Saved at models/model.pkl")
print("Label encoders saved at models/label_encoders.pkl")
print("Preprocessing metadata saved at models/preprocessing.pkl")
