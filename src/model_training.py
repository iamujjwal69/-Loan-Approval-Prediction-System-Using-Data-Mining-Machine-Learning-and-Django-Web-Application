"""
Machine Learning Model Training & Comparative Benchmarking Module.
Trains Logistic Regression, Decision Tree, and Random Forest models encapsulated
within full Scikit-Learn Pipelines. Compares cross-validation scores and persists the champion model.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BEST_PIPELINE_PATH,
    CV_FOLDS,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.model_evaluation import evaluate_model
from src.preprocessing import build_preprocessor


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Returns candidate classification algorithms with tuned hyperparameters.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=6,
            min_samples_leaf=3,
            criterion="gini",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
        ),
    }


def train_and_benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    persist_best: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], str, Pipeline]:
    """
    Trains all candidate classifiers within encapsulated pipelines, computes
    stratified 5-fold cross-validation, evaluates on the held-out test split,
    and identifies the best model based on F1-Score and Accuracy.
    """
    candidate_estimators = get_candidate_models(RANDOM_STATE)
    benchmarks: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_score = -1.0
    best_model_name = ""
    best_pipeline = None

    for model_name, classifier in candidate_estimators.items():
        # Encapsulate preprocessing + classifier into an immutable pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        # Stratified Cross-Validation on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        # Fit on entire training split
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        # Comprehensive test set evaluation
        metrics = evaluate_model(pipeline, X_test, y_test, model_name=model_name)
        metrics["cv_mean_accuracy"] = round(cv_mean, 4)
        metrics["cv_std_accuracy"] = round(cv_std, 4)
        metrics["hyperparameters"] = classifier.get_params()

        benchmarks[model_name] = metrics

        # Select model prioritizing balanced F1-score and generalization accuracy
        composite_score = (metrics["f1_score"] * 0.6) + (metrics["accuracy"] * 0.4)
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = model_name
            best_pipeline = pipeline

    # Flag active best model in benchmarks
    for m_name in benchmarks:
        benchmarks[m_name]["is_best"] = (m_name == best_model_name)

    # Persistence
    if persist_best and best_pipeline is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_pipeline, BEST_PIPELINE_PATH)

        metadata = {
            "best_model_name": best_model_name,
            "trained_at": datetime.now().isoformat(),
            "features": MODEL_FEATURES,
            "test_metrics": benchmarks[best_model_name],
            "all_model_benchmarks": {k: {m: v for m, v in val.items() if m != "hyperparameters"} for k, val in benchmarks.items()},
        }

        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)

        with open(REPORTS_DIR / "model_comparison_summary.json", "w") as f:
            json.dump(benchmarks, f, indent=4)

    return benchmarks, best_model_name, best_pipeline
