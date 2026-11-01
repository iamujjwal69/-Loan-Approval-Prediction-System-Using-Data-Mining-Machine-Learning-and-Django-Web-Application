"""
Feature Selection & Importance Ranking Module.
Analyzes feature relevance using tree-based feature importance
and outputs high-DPI importance bar charts for the analytics dashboard.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import VISUALIZATIONS_DIR
from src.preprocessing import build_preprocessor


def compute_feature_importances(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Path]:
    """
    Fits preprocessor and a baseline RandomForestClassifier to compute Gini feature importances.
    Generates and saves the feature importance visualization.
    """
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Extract human-readable feature names from ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()
    # Clean up prefixes: e.g. "num__ApplicantIncome" -> "ApplicantIncome"
    clean_names = [name.split("__")[-1] for name in feature_names]

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
    rf.fit(X_train_transformed, y_train)

    importances = rf.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": clean_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Plot and save
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues_r(np_linspace := [i / len(importance_df) for i in range(len(importance_df))])
    
    bars = ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#2563eb", edgecolor="#1e3a8a")
    ax.set_title("Random Forest Gini Feature Importance Ranking", fontweight="bold", pad=15)
    ax.set_xlabel("Relative Importance Score")
    ax.set_ylabel("Predictor Features")

    # Add score annotations
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(importances) * 1.15)
    
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VISUALIZATIONS_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return importance_df, out_path
