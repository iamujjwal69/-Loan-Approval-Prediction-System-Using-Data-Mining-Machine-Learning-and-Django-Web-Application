"""
Data Loader & Dataset Ingestion Module.
Handles CSV and Excel file ingestion, schema validation, and summary inspection.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def load_dataset(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load dataset from CSV or XLSX file. Defaults to baseline raw dataset if not provided.
    """
    if file_path is None:
        file_path = RAW_DATA_PATH

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .csv, .xlsx, .xls")

    return df


def validate_dataset_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str]]:
    """
    Validates whether the DataFrame contains necessary features and target.
    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if df.empty:
        errors.append("Dataset is empty.")
        return False, errors

    # Check for target column
    if require_target and TARGET_COLUMN not in df.columns:
        errors.append(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

    # Check for expected input features
    missing_features = [col for col in MODEL_FEATURES if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {', '.join(missing_features)}")

    is_valid = len(errors) == 0
    return is_valid, errors


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics for dataset preview and dashboard display.
    """
    total_records = len(df)
    total_columns = len(df.columns)
    duplicate_count = int(df.duplicated().sum())

    missing_series = df.isnull().sum()
    missing_counts = {col: int(cnt) for col, cnt in missing_series.items() if cnt > 0}
    total_missing_cells = int(missing_series.sum())

    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    num_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    target_distribution: Dict[str, int] = {}
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts().to_dict()
        target_distribution = {str(k): int(v) for k, v in counts.items()}

    # Preview records (clean NaN to None for JSON serialization)
    preview_df = df.head(10).copy()
    raw_records = preview_df.to_dict(orient="records")
    clean_records = []
    for rec in raw_records:
        clean_rec = {}
        for k, v in rec.items():
            if pd.isna(v):
                clean_rec[k] = None
            else:
                clean_rec[k] = v
        clean_records.append(clean_rec)

    return {
        "total_records": total_records,
        "total_columns": total_columns,
        "total_missing_cells": total_missing_cells,
        "missing_counts": missing_counts,
        "duplicate_count": duplicate_count,
        "dtypes": dtypes_dict,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "target_distribution": target_distribution,
        "preview_records": clean_records,
        "columns": list(df.columns),
    }
