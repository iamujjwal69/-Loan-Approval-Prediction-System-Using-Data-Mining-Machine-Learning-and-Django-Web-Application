"""
Data Cleaning & Missing Value Handling Module.
Provides functions for duplicate detection, null-value analysis,
data type correction, and exploratory imputation.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing missing values per column with percentages.
    """
    total_missing = df.isnull().sum()
    percent_missing = (total_missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing_Count": total_missing.values,
            "Percentage": percent_missing.round(2).values,
            "Dtype": [str(d) for d in df.dtypes],
        }
    )
    return missing_df[missing_df["Missing_Count"] > 0].sort_values(
        by="Missing_Count", ascending=False
    ).reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identifies and removes duplicate rows from the dataset.
    Returns (cleaned_df, duplicate_count).
    """
    initial_count = len(df)
    cleaned_df = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(cleaned_df)
    return cleaned_df, duplicates_removed


def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes string columns by stripping leading/trailing whitespace
    and standardizing common formatting anomalies.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        # Restore actual NaNs that became the string 'nan'
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Standardize Dependents: '3+' is kept as '3+' or string category
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace({"3+": "3+"})

    return df


def impute_missing_exploratory(
    df: pd.DataFrame,
    num_strategy: str = "median",
    cat_strategy: str = "mode",
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Imputes missing values for exploratory analysis and preview.
    - Numerical: Mean or Median
    - Categorical: Mode (most frequent)
    Returns (imputed_df, imputation_values_dict).
    """
    df_imputed = df.copy()
    imputation_records: Dict[str, any] = {}

    # Numerical imputation
    for col in NUMERICAL_FEATURES:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            if num_strategy == "median":
                fill_val = df_imputed[col].median()
            else:
                fill_val = df_imputed[col].mean()
            df_imputed[col] = df_imputed[col].fillna(fill_val)
            imputation_records[col] = fill_val

    # Categorical imputation
    for col in CATEGORICAL_FEATURES:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            mode_val = df_imputed[col].mode(dropna=True)[0]
            df_imputed[col] = df_imputed[col].fillna(mode_val)
            imputation_records[col] = mode_val

    return df_imputed, imputation_records
