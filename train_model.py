"""
Run this script ONCE to train the model and save it.
Usage: python train_model.py
Requires: credit_risk_dataset.csv in the same folder
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

# ── Load data ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(BASE_DIR, "credit_risk_dataset.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {CSV_PATH}.\n"
        "Please place 'credit_risk_dataset.csv' in the same directory as this script."
    )

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Preprocessing ──────────────────────────────────────────────────────────────
# Fill missing Income
df["Income"] = df["Income"].fillna(df["Income"].mean())

# Encode Education_Level
edu_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
df["Education_Level"] = df["Education_Level"].map(edu_order)

# One-hot encode Housing_Status (drop_first keeps Own as baseline)
df = pd.get_dummies(df, columns=["Housing_Status"], drop_first=True, dtype=int)

FEATURE_NAMES = [
    "Age", "Income", "Loan_Amount", "Credit_Score",
    "Employment_Years", "Education_Level",
    "Housing_Status_Own", "Housing_Status_Rent",
]

X = df[FEATURE_NAMES]
y = df["Default"]

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / Test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=10
)

# ── Train Logistic Regression ──────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = lr.predict(X_test)
print("\nModel Performance:")
print(f"Accuracy : {lr.score(X_test, y_test):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save artifacts ─────────────────────────────────────────────────────────────
joblib.dump(lr, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
joblib.dump(FEATURE_NAMES, os.path.join(BASE_DIR, "feature_names.pkl"))
print("\nSaved model.pkl, scaler.pkl, and feature_names.pkl")
