"""
Central Configuration Module for Loan Approval Prediction System.
Defines directory paths, dataset schemas, valid category values, and model hyperparameters.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "loan_data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_preprocessed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_preprocessed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_PIPELINE_PATH = MODELS_DIR / "best_loan_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_SUMMARY_PATH = REPORTS_DIR / "model_comparison_summary.json"

# Dataset Schema
ID_COLUMN = "Loan_ID"
TARGET_COLUMN = "Loan_Status"

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERICAL_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Combined input features used in modeling & web form
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Domain mappings and category rules
CATEGORICAL_CHOICES = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "Property_Area": ["Urban", "Semiurban", "Rural"],
    "Credit_History": [1.0, 0.0],
}

TARGET_MAPPING = {"Y": 1, "N": 0}
TARGET_INVERSE_MAPPING = {1: "Y", 0: "N"}
STATUS_LABELS = {1: "Approved", 0: "Rejected"}

# Training Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
