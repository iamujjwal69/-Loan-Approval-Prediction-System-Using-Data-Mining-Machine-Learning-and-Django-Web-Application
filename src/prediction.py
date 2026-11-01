"""
Real-time Loan Prediction & Inference Module.
Loads the serialized Scikit-Learn pipeline, validates raw applicant feature inputs,
and outputs predicted approval class, calibrated probabilities, and risk ratings.
"""

from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd

from src.config import (
    BEST_PIPELINE_PATH,
    MODEL_FEATURES,
    STATUS_LABELS,
    TARGET_INVERSE_MAPPING,
)

# Global cached pipeline
_CACHED_PIPELINE = None


def get_inference_pipeline():
    """
    Returns the loaded pipeline artifact with memory caching.
    """
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not BEST_PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {BEST_PIPELINE_PATH}. Run training first."
            )
        _CACHED_PIPELINE = joblib.load(BEST_PIPELINE_PATH)
    return _CACHED_PIPELINE


def predict_loan_approval(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs inference for a single loan applicant.
    
    Expected applicant_data format:
    {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000.0,
        "CoapplicantIncome": 1500.0,
        "LoanAmount": 150.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }
    """
    pipeline = get_inference_pipeline()

    # Convert dictionary to single-row DataFrame with expected columns
    input_df = pd.DataFrame([applicant_data])

    # Validate that all expected features are present
    for feature in MODEL_FEATURES:
        if feature not in input_df.columns:
            raise ValueError(f"Missing required input feature: {feature}")

    # Reorder columns to match training schema
    input_df = input_df[MODEL_FEATURES]

    # Predict class (0 or 1)
    raw_pred = pipeline.predict(input_df)[0]
    predicted_status = TARGET_INVERSE_MAPPING[int(raw_pred)]  # 'Y' or 'N'
    status_label = STATUS_LABELS[int(raw_pred)]  # 'Approved' or 'Rejected'

    # Predict probabilities
    prob_approved = 0.5
    prob_rejected = 0.5
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        # Class 0: Rejected, Class 1: Approved
        prob_rejected = float(probs[0])
        prob_approved = float(probs[1])

    # Determine qualitative Risk Assessment Level
    if prob_approved >= 0.75:
        risk_level = "Low Risk"
        risk_badge = "success"
    elif prob_approved >= 0.50:
        risk_level = "Moderate Risk"
        risk_badge = "warning"
    else:
        risk_level = "High Risk"
        risk_badge = "danger"

    return {
        "predicted_result": predicted_status,
        "status_label": status_label,
        "probability_approved": round(prob_approved * 100, 2),
        "probability_rejected": round(prob_rejected * 100, 2),
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "applicant_input": applicant_data,
    }
