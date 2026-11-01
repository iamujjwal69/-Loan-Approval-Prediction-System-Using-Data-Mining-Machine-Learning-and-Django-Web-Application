"""
Scikit-Learn Preprocessing Pipeline & ColumnTransformer Module.
Encapsulates missing value imputation, standard scaling, and one-hot encoding
strictly within Pipeline components to prevent data leakage between train and test sets.
"""

from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an encapsulated ColumnTransformer for tabular preprocessing:
    - Numerical: Median imputation -> StandardScaler
    - Categorical: Mode (most frequent) imputation -> OneHotEncoder
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into stratified Train and Test partitions.
    Ensures that transformations are NEVER fitted on test data.
    """
    # Clean string features
    clean_df = df.copy()
    if ID_COLUMN in clean_df.columns:
        clean_df = clean_df.drop(columns=[ID_COLUMN])

    X = clean_df[MODEL_FEATURES].copy()
    
    # Target encoding: 'Y' -> 1, 'N' -> 0
    if TARGET_COLUMN in clean_df.columns:
        y_raw = clean_df[TARGET_COLUMN].map(TARGET_MAPPING)
        # Drop rows where target is missing if any
        valid_idx = y_raw.notnull()
        X = X.loc[valid_idx]
        y = y_raw.loc[valid_idx].astype(int)
    else:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required for training split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
