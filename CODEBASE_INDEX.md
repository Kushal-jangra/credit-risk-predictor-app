# Codebase Index

## Overview

This repository is a compact credit risk prediction project with two active code paths:

- model training in `training/train.py`
- model serving in `api/main.py`

The rest of the repository is primarily data, generated model artifacts, and a checked-in Python virtual environment.

## Top-Level Layout

### `/Users/kushal/Project_5/api`

- `main.py`
  - FastAPI application entry point
  - Loads persisted assets from `models/`
  - Normalizes request payloads into model feature order
  - Validates categorical values against fitted label encoders
  - Exposes `GET /` and `POST /predict`

### `/Users/kushal/Project_5/training`

- `train.py`
  - End-to-end training script
  - Reads the CSV dataset from `data/credit_risk_dataset.csv`
  - Fills missing numeric values with medians
  - Fills missing categorical values with `"Unknown"`
  - Label-encodes categorical columns
  - Trains a `RandomForestClassifier`
  - Evaluates with holdout accuracy
  - Persists model and preprocessing assets into `models/`

### `/Users/kushal/Project_5/data`

- `credit_risk_dataset.csv`
  - Main tabular dataset used for training
  - Feature columns:
    - `person_age`
    - `person_income`
    - `person_home_ownership`
    - `person_emp_length`
    - `loan_intent`
    - `loan_grade`
    - `loan_amnt`
    - `loan_int_rate`
    - `loan_percent_income`
    - `cb_person_default_on_file`
    - `cb_person_cred_hist_length`
  - Target column:
    - `loan_status`

### `/Users/kushal/Project_5/models`

- `model.pkl`
  - Trained scikit-learn model used by the API
- `label_encoders.pkl`
  - Per-column fitted `LabelEncoder` objects for categorical inputs
- `preprocessing.pkl`
  - Saved feature order, categorical column list, and numeric fill defaults

### `/Users/kushal/Project_5/venv`

- Local Python virtual environment
- Contains third-party packages and executables
- Not part of the application source itself
- Should generally be excluded from code search, indexing, and review

## Runtime Entry Points

- Training: `python training/train.py`
- API: `uvicorn api.main:app --reload`

## Request Flow

1. `api/main.py` loads the trained artifacts from `models/`
2. `POST /predict` accepts a raw JSON dictionary
3. `preprocess_input()` orders features to match training
4. Categorical inputs are validated and label-encoded
5. Numeric inputs are cast to float and backfilled when missing
6. The model predicts `loan_status`
7. The API maps the numeric prediction to `High Risk` or `Low Risk`

## Training Outputs

Running `training/train.py` produces or refreshes:

- `models/model.pkl`
- `models/label_encoders.pkl`
- `models/preprocessing.pkl`

These artifacts are required for the API to start successfully.

## Key Dependencies

- `pandas`
- `numpy`
- `joblib`
- `fastapi`
- `scikit-learn`
- `uvicorn`

## Notes

- The repository is not currently a Git repository from this directory, so Git-based indexing metadata is unavailable here.
- `api/__pycache__/` and the entire `models/` directory are generated artifacts rather than hand-authored source.
- The dominant file count comes from `venv/`, not from project code.
