"""
FastAPI Loan Defaulter Prediction App
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # abspath fixes Render CWD issues
MODEL_PATH         = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH        = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")
TEMPLATES_DIR      = os.path.join(BASE_DIR, "templates")
STATIC_DIR         = os.path.join(BASE_DIR, "static")

# ── Validate artifacts ─────────────────────────────────────────────────────────
for path, name in [
    (MODEL_PATH,         "model.pkl"),
    (SCALER_PATH,        "scaler.pkl"),
    (FEATURE_NAMES_PATH, "feature_names.pkl"),
]:
    if not os.path.exists(path):
        raise RuntimeError(
            f"{name} not found at {path}.\n"
            "Please run  python train_model.py  first."
        )

model         = joblib.load(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)

# ── FastAPI setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Loan Defaulter Predictor")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ── Encodings (must match training) ───────────────────────────────────────────
EDU_MAP = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
HOUSING_MAP = {
    "Mortgage": (0, 0),
    "Own":      (1, 0),
    "Rent":     (0, 1),
}


def _risk_level(prob: float) -> str:
    if prob < 30:  return "Low"
    if prob < 60:  return "Medium"
    if prob < 80:  return "High"
    return "Very High"


def _build_features(age, income, loan_amount, credit_score,
                    employment_years, education_level, housing_status):
    edu_encoded = EDU_MAP.get(education_level, 1)
    housing_own, housing_rent = HOUSING_MAP.get(housing_status, (0, 0))
    features_df = pd.DataFrame([[
        age, income, loan_amount, credit_score,
        employment_years, edu_encoded, housing_own, housing_rent,
    ]], columns=FEATURE_NAMES)
    return scaler.transform(features_df)


# ── Web routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "form_data": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float              = Form(...),
    income: float           = Form(...),
    loan_amount: float      = Form(...),
    credit_score: float     = Form(...),
    employment_years: float = Form(...),
    education_level: str    = Form(...),
    housing_status: str     = Form(...),
):
    features_scaled = _build_features(
        age, income, loan_amount, credit_score,
        employment_years, education_level, housing_status
    )
    prediction   = model.predict(features_scaled)[0]
    proba        = model.predict_proba(features_scaled)[0]
    default_prob = round(float(proba[1]) * 100, 2)
    safe_prob    = round(float(proba[0]) * 100, 2)

    result = {
        "is_defaulter": bool(prediction),
        "default_prob": default_prob,
        "safe_prob":    safe_prob,
        "label":        "⚠️ LIKELY DEFAULTER" if prediction else "✅ NOT A DEFAULTER",
        "risk_level":   _risk_level(default_prob),
    }
    form_data = {
        "age": age, "income": income, "loan_amount": loan_amount,
        "credit_score": credit_score, "employment_years": employment_years,
        "education_level": education_level, "housing_status": housing_status,
    }
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result, "form_data": form_data}
    )


# ── REST API ───────────────────────────────────────────────────────────────────
from pydantic import BaseModel

class CustomerData(BaseModel):
    age: float
    income: float
    loan_amount: float
    credit_score: float
    employment_years: float
    education_level: str
    housing_status: str


@app.post("/api/predict")
async def api_predict(data: CustomerData):
    features_scaled = _build_features(
        data.age, data.income, data.loan_amount, data.credit_score,
        data.employment_years, data.education_level, data.housing_status
    )
    prediction   = int(model.predict(features_scaled)[0])
    proba        = model.predict_proba(features_scaled)[0]
    default_prob = round(float(proba[1]) * 100, 2)
    return {
        "prediction":          prediction,
        "is_defaulter":        bool(prediction),
        "default_probability": default_prob,
        "safe_probability":    round(float(proba[0]) * 100, 2),
        "risk_level":          _risk_level(default_prob),
    }
