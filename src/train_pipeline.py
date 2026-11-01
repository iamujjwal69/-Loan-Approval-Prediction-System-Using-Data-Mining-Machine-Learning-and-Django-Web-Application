"""
End-to-End Pipeline Execution & Training Script.
Orchestrates EDA visualization generation, stratified splitting, model benchmarking,
feature importance ranking, pipeline persistence, and sample real-time inference.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH
from src.data_cleaning import clean_categorical_values, remove_duplicates
from src.data_loader import load_dataset
from src.eda import generate_all_visualizations
from src.feature_selection import compute_feature_importances
from src.model_training import train_and_benchmark_models
from src.outlier_detection import cap_outliers_iqr
from src.prediction import predict_loan_approval
from src.preprocessing import prepare_train_test_data


def run_pipeline():
    print("=" * 80)
    print("LOAN APPROVAL PREDICTION SYSTEM: FULL MACHINE LEARNING PIPELINE")
    print("=" * 80)

    # 1. Ingestion
    print("\n[Step 1] Loading raw dataset from:", RAW_DATA_PATH)
    df = load_dataset(RAW_DATA_PATH)
    print(f" -> Ingested {len(df)} records, {len(df.columns)} columns.")

    # 2. Cleaning & Outlier Treatment
    print("\n[Step 2] Cleaning records and capping outliers...")
    clean_df, dupes = remove_duplicates(df)
    clean_df = clean_categorical_values(clean_df)
    clean_df, _ = cap_outliers_iqr(clean_df)
    print(" -> Data sanitization and IQR capping complete.")

    # 3. Exploratory Data Analysis & Plots
    print("\n[Step 3] Generating publication-grade visualizations...")
    vis_paths = generate_all_visualizations(clean_df)
    for name, p in vis_paths.items():
        print(f"    * {name}: {p}")

    # 4. Stratified Train / Test Partitioning
    print("\n[Step 4] Performing Stratified Train-Test Splitting (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(clean_df)
    print(f" -> X_train shape: {X_train.shape}, y_train class balance: {dict(y_train.value_counts())}")
    print(f" -> X_test shape:  {X_test.shape},  y_test class balance:  {dict(y_test.value_counts())}")

    # 5. Feature Importance Analysis
    print("\n[Step 5] Computing Feature Importances...")
    imp_df, imp_plot = compute_feature_importances(X_train, y_train)
    print(" -> Top 5 Most Important Features:")
    print(imp_df.head(5).to_string(index=False))

    # 6. Model Benchmarking & Pipeline Serialization
    print("\n[Step 6] Benchmarking Candidate Classifiers...")
    benchmarks, best_name, _ = train_and_benchmark_models(
        X_train, y_train, X_test, y_test, persist_best=True
    )

    print("\n" + "-" * 75)
    print(f"{'Model':<24} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'CV Mean':<10}")
    print("-" * 75)
    for m_name, metrics in benchmarks.items():
        star = " *" if metrics.get("is_best") else ""
        print(
            f"{m_name + star:<24} | "
            f"{metrics['accuracy']:<10.4f} | "
            f"{metrics['precision']:<10.4f} | "
            f"{metrics['recall']:<10.4f} | "
            f"{metrics['f1_score']:<10.4f} | "
            f"{metrics['cv_mean_accuracy']:<10.4f}"
        )
    print("-" * 75)
    print(f"Selected Champion Model: '{best_name}' (Persisted with full preprocessor to models/)")

    # 7. Verification of Production Inference
    print("\n[Step 7] Testing Real-Time Prediction Inference Engine...")
    sample_applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 4583.0,
        "CoapplicantIncome": 1508.0,
        "LoanAmount": 128.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban",
    }
    pred_result = predict_loan_approval(sample_applicant)
    print(f" -> Predicted Status:       {pred_result['status_label']} ({pred_result['predicted_result']})")
    print(f" -> Approval Probability:   {pred_result['probability_approved']}%")
    print(f" -> Rejection Probability:  {pred_result['probability_rejected']}%")
    print(f" -> Risk Assessment:       {pred_result['risk_level']}")

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
