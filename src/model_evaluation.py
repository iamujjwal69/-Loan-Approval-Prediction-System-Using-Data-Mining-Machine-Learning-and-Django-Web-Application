"""
Model Evaluation & Performance Metrics Module.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix,
and full Classification Reports. Generates high-DPI Confusion Matrix plots.
"""

from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import VISUALIZATIONS_DIR


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier pipeline on test data and returns a comprehensive metrics dictionary.
    """
    y_pred = model.predict(X_test)
    y_prob = None
    roc_auc = None

    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            pass

    acc = round(float(accuracy_score(y_test, y_pred)), 4)
    prec = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()  # [[TN, FP], [FN, TP]]

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"], zero_division=0)

    # Generate and save confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected (0)", "Approved (1)"],
        yticklabels=["Rejected (0)", "Approved (1)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(" ", "_")
    cm_plot_path = VISUALIZATIONS_DIR / f"confusion_matrix_{slug}.png"
    fig.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm_list,
        "confusion_matrix_plot": str(cm_plot_path),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }
