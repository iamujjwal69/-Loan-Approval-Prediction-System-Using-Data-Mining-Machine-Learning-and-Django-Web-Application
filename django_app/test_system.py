"""
Comprehensive Academic Test Suite for Loan Approval Prediction System.
Validates all 11 test cases specified in the project engineering requirements:
1. Valid CSV upload
2. Valid XLSX upload
3. Invalid file upload
4. Missing value handling
5. Duplicate removal
6. Outlier detection
7. Model training & serialization
8. Prediction with valid input
9. Prediction with invalid input
10. Unauthorized dashboard access
11. Prediction history storage
"""

import io
import json
import os
import sys
from pathlib import Path
import django
from django.test import Client, TestCase

# Initialize Django environment
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
import pandas as pd
from datasets.models import Dataset
from predictions.models import ModelPerformance, PredictionHistory
from src.config import BEST_PIPELINE_PATH, RAW_DATA_PATH
from src.data_cleaning import get_missing_summary, impute_missing_exploratory, remove_duplicates
from src.data_loader import load_dataset, validate_dataset_schema
from src.outlier_detection import cap_outliers_iqr, detect_outliers_iqr
from src.prediction import predict_loan_approval


def run_all_tests():
    print("=" * 80)
    print("LOAN APPROVAL PREDICTION SYSTEM: VERIFICATION OF ALL 11 TEST CASES")
    print("=" * 80)

    client = Client()

    # Create test user
    test_user, _ = User.objects.get_or_create(username="test_officer")
    test_user.set_password("pass12345")
    test_user.save()

    # --- Test Case 1: Valid CSV Upload ---
    print("\n[Test 1] Valid CSV upload...")
    client.force_login(test_user)
    with open(RAW_DATA_PATH, "rb") as f:
        csv_file = SimpleUploadedFile("test_loans.csv", f.read(), content_type="text/csv")
    resp = client.post("/datasets/upload/", {"dataset_name": "Test CSV Upload", "file": csv_file}, follow=True)
    assert resp.status_code == 200
    assert Dataset.objects.filter(dataset_name="Test CSV Upload").exists()
    print(" -> PASSED: Valid CSV accepted, metadata registered, status 200.")

    # --- Test Case 2: Valid XLSX Upload ---
    print("\n[Test 2] Valid XLSX upload...")
    df_sample = load_dataset(RAW_DATA_PATH).head(25)
    xlsx_buffer = io.BytesIO()
    df_sample.to_excel(xlsx_buffer, index=False)
    xlsx_buffer.seek(0)
    xlsx_file = SimpleUploadedFile("sample_excel.xlsx", xlsx_buffer.read(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    resp_xlsx = client.post("/datasets/upload/", {"dataset_name": "Test XLSX Upload", "file": xlsx_file}, follow=True)
    assert resp_xlsx.status_code == 200
    assert Dataset.objects.filter(dataset_name="Test XLSX Upload").exists()
    print(" -> PASSED: Valid XLSX parsed and registered successfully.")

    # --- Test Case 3: Invalid File Upload ---
    print("\n[Test 3] Invalid file upload (wrong extension / invalid file)...")
    bad_file = SimpleUploadedFile("malicious.exe", b"binary content", content_type="application/octet-stream")
    resp_bad = client.post("/datasets/upload/", {"dataset_name": "Bad File", "file": bad_file}, follow=True)
    assert not Dataset.objects.filter(dataset_name="Bad File").exists()
    print(" -> PASSED: Rejected invalid extension (.exe) with form validation error.")

    # --- Test Case 4: Missing Value Handling ---
    print("\n[Test 4] Missing value detection and imputation...")
    raw_df = load_dataset(RAW_DATA_PATH)
    missing_summary = get_missing_summary(raw_df)
    assert len(missing_summary) > 0, "Missing values should be detected in raw data"
    imputed_df, fill_vals = impute_missing_exploratory(raw_df)
    remaining_nulls = imputed_df[["ApplicantIncome", "LoanAmount", "Credit_History"]].isnull().sum().sum()
    assert remaining_nulls == 0
    print(f" -> PASSED: Missing values identified ({len(missing_summary)} columns) and imputed (remaining nulls: {remaining_nulls}).")

    # --- Test Case 5: Duplicate Removal ---
    print("\n[Test 5] Duplicate record detection and removal...")
    dupe_df = pd.concat([raw_df.head(10), raw_df.head(10)], ignore_index=True)
    cleaned_dupes, count = remove_duplicates(dupe_df)
    assert count == 10
    assert len(cleaned_dupes) == 10
    print(f" -> PASSED: Detected and removed exactly {count} duplicated rows.")

    # --- Test Case 6: Outlier Detection ---
    print("\n[Test 6] Outlier detection via IQR method...")
    outliers = detect_outliers_iqr(raw_df, "ApplicantIncome", factor=1.5)
    assert outliers["outlier_count"] > 0
    assert outliers["upper_bound"] > 0
    capped_df, _ = cap_outliers_iqr(raw_df, columns=["ApplicantIncome"])
    assert capped_df["ApplicantIncome"].max() <= outliers["upper_bound"]
    print(f" -> PASSED: IQR boundaries computed [{outliers['lower_bound']}, {outliers['upper_bound']}], {outliers['outlier_count']} outliers capped.")

    # --- Test Case 7: Model Training & Persistence ---
    print("\n[Test 7] Model training and pipeline persistence verification...")
    assert BEST_PIPELINE_PATH.exists(), f"Saved pipeline artifact missing at {BEST_PIPELINE_PATH}"
    assert ModelPerformance.objects.count() >= 3
    print(" -> PASSED: Serialized Scikit-Learn pipeline exists on disk and 3 model benchmarks populated.")

    # --- Test Case 8: Prediction with Valid Input ---
    print("\n[Test 8] Real-time inference with valid applicant data...")
    valid_applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "2",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 6000.0,
        "CoapplicantIncome": 2000.0,
        "LoanAmount": 140.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban",
    }
    result = predict_loan_approval(valid_applicant)
    assert result["predicted_result"] in ["Y", "N"]
    assert 0.0 <= result["probability_approved"] <= 100.0
    print(f" -> PASSED: Valid applicant predicted as '{result['status_label']}' (Approval Prob: {result['probability_approved']}%, Risk: {result['risk_level']}).")

    # --- Test Case 9: Prediction with Invalid Input ---
    print("\n[Test 9] Prediction error handling with invalid / missing features...")
    invalid_applicant = dict(valid_applicant)
    del invalid_applicant["ApplicantIncome"]  # Strip required feature
    threw_error = False
    try:
        predict_loan_approval(invalid_applicant)
    except ValueError:
        threw_error = True
    assert threw_error
    print(" -> PASSED: ValueError caught properly when mandatory feature is missing.")

    # --- Test Case 10: Unauthorized Dashboard Access ---
    print("\n[Test 10] Unauthorized dashboard access security guard...")
    anonymous_client = Client()
    resp_unauth = anonymous_client.get("/")
    assert resp_unauth.status_code == 302
    assert "/accounts/login/" in resp_unauth.url
    print(" -> PASSED: Unauthenticated user redirected to /accounts/login/ (Status 302).")

    # --- Test Case 11: Prediction History Storage ---
    print("\n[Test 11] Auditing and storage of prediction records in SQLite...")
    initial_count = PredictionHistory.objects.count()
    resp_pred = client.post(
        "/predict/",
        {
            "gender": "Female",
            "married": "No",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 4800,
            "coapplicant_income": 0,
            "loan_amount": 110,
            "loan_amount_term": 360,
            "credit_history": 1.0,
            "property_area": "Urban",
        },
        follow=True,
    )
    assert resp_pred.status_code == 200
    assert PredictionHistory.objects.count() == initial_count + 1
    latest_rec = PredictionHistory.objects.first()
    assert latest_rec.user == test_user
    print(f" -> PASSED: Prediction successfully persisted to database (ID #{latest_rec.id}, Result: {latest_rec.status_label}).")

    print("\n" + "=" * 80)
    print("ALL 11 ACADEMIC TEST CASES COMPLETED AND PASSED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
