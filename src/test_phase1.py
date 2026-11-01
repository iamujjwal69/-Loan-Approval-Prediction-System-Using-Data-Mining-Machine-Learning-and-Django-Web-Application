"""
Phase 1 Verification Script:
Tests and asserts Data Ingestion, Schema Validation, Data Cleaning, and Outlier Handling.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_cleaning import (
    clean_categorical_values,
    get_missing_summary,
    impute_missing_exploratory,
    remove_duplicates,
)
from src.data_loader import (
    get_dataset_overview,
    load_dataset,
    validate_dataset_schema,
)
from src.outlier_detection import (
    cap_outliers_iqr,
    detect_outliers_iqr,
    get_all_outliers_summary,
)


def run_phase1_validation():
    print("=" * 70)
    print("PHASE 1 VALIDATION: DATA MINING & INGESTION PIPELINE")
    print("=" * 70)

    # 1. Dataset Loading
    print("\n[1] Loading Baseline Dataset...")
    df = load_dataset(RAW_DATA_PATH)
    print(f" -> Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    assert len(df) == 614, f"Expected 614 rows, got {len(df)}"
    assert len(df.columns) == 13, f"Expected 13 columns, got {len(df.columns)}"

    # 2. Schema Validation
    print("\n[2] Validating Schema Integrity...")
    is_valid, errors = validate_dataset_schema(df, require_target=True)
    assert is_valid, f"Schema validation failed: {errors}"
    print(f" -> Schema valid: All {len(MODEL_FEATURES)} model features + target '{TARGET_COLUMN}' present.")

    # 3. Overview Inspection
    print("\n[3] Generating Dataset Overview Metrics...")
    overview = get_dataset_overview(df)
    print(f" -> Total Records: {overview['total_records']}")
    print(f" -> Total Columns: {overview['total_columns']}")
    print(f" -> Total Missing Cells: {overview['total_missing_cells']}")
    print(f" -> Duplicate Records: {overview['duplicate_count']}")
    print(f" -> Target Distribution: {overview['target_distribution']}")

    # 4. Missing Values Analysis
    print("\n[4] Analyzing Missing Values...")
    missing_summary = get_missing_summary(df)
    print(missing_summary.to_string(index=False))

    # 5. Duplicate Check & Cleaning
    print("\n[5] Checking and Sanitizing Records...")
    cleaned_df, dupes = remove_duplicates(df)
    print(f" -> Duplicates removed: {dupes}")
    cleaned_df = clean_categorical_values(cleaned_df)
    print(" -> Categorical text sanitization completed.")

    # 6. Outlier Detection
    print("\n[6] Detecting Outliers via IQR Method (Factor = 1.5)...")
    outliers_table = get_all_outliers_summary(cleaned_df)
    print(outliers_table.to_string(index=False))

    # 7. Outlier Capping (Winsorization)
    print("\n[7] Applying IQR Capping (Winsorization)...")
    capped_df, bounds = cap_outliers_iqr(cleaned_df)
    for col, b in bounds.items():
        print(f" -> {col}: Lower Bound = {b['lower_bound']}, Upper Bound = {b['upper_bound']}")

    # 8. Exploratory Imputation Check
    print("\n[8] Testing Exploratory Imputation...")
    imputed_df, filled_values = impute_missing_exploratory(capped_df)
    remaining_nulls = imputed_df[MODEL_FEATURES].isnull().sum().sum()
    print(f" -> Remaining nulls in features after imputation: {remaining_nulls}")
    assert remaining_nulls == 0, "Imputation should resolve all nulls in model features."
    print(" -> Imputation Values Sample:")
    for k, v in list(filled_values.items())[:5]:
        print(f"    * {k}: {v}")

    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION SUCCESSFUL: ALL CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1_validation()
