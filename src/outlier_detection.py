"""
Outlier Detection & Capping Module.
Implements Interquartile Range (IQR) outlier detection and domain-safe
capping (Winsorization) to preserve valid sample observations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import NUMERICAL_FEATURES


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    factor: float = 1.5,
) -> Dict[str, any]:
    """
    Calculates IQR boundaries and identifies outliers for a specific numerical column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    series = df[column].dropna()
    if series.empty:
        return {
            "column": column,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
        }

    q1 = float(np.percentile(series, 25))
    q3 = float(np.percentile(series, 75))
    iqr = q3 - q1

    lower_bound = float(q1 - (factor * iqr))
    upper_bound = float(q3 + (factor * iqr))

    # Real-world constraint: financial features like income/loan cannot be negative
    lower_bound = max(0.0, lower_bound)

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = round((outlier_count / len(series)) * 100, 2)

    return {
        "column": column,
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_pct,
    }


def get_all_outliers_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Scans all specified numerical columns and outputs a consolidated outlier summary table.
    """
    if columns is None:
        columns = [col for col in NUMERICAL_FEATURES if col in df.columns and col != "Credit_History"]

    results = []
    for col in columns:
        stats = detect_outliers_iqr(df, col, factor=factor)
        results.append(stats)

    return pd.DataFrame(results)


def cap_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Caps extreme outlier values to IQR lower and upper bounds (Winsorization).
    Avoids data loss caused by row deletion while preserving statistical validity.
    """
    df_capped = df.copy()
    if columns is None:
        columns = [col for col in NUMERICAL_FEATURES if col in df_capped.columns and col != "Credit_History"]

    bounds_applied: Dict[str, Dict[str, float]] = {}

    for col in columns:
        stats = detect_outliers_iqr(df_capped, col, factor=factor)
        lower = stats["lower_bound"]
        upper = stats["upper_bound"]

        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        bounds_applied[col] = {"lower_bound": lower, "upper_bound": upper}

    return df_capped, bounds_applied
