"""
Database Seeder Script.
Populates SQLite with trained model benchmark records, default datasets,
and creates the administrator account for project presentation.
"""

import json
import os
import sys
from pathlib import Path
import django

# Setup Django environment
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User
from datasets.models import Dataset
from predictions.models import ModelPerformance, PredictionHistory
from src.config import METRICS_SUMMARY_PATH, RAW_DATA_PATH
from src.data_loader import get_dataset_overview, load_dataset


def sanitize_for_json(data):
    """Recursively replaces NaN/Infinity and non-standard objects with JSON-safe values."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, float):
        import math
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


def seed():
    print("=" * 60)
    print("SEEDING DATABASE WITH TRAINED METRICS AND ACCOUNTS")
    print("=" * 60)

    # 1. Superuser Creation
    if not User.objects.filter(username="admin").exists():
        admin = User.objects.create_superuser("admin", "admin@loananalytics.edu", "admin123")
        print(" -> Created Admin user: username='admin', password='admin123'")
    else:
        admin = User.objects.get(username="admin")
        print(" -> Admin user already exists.")

    # 2. Seed Model Performance Metrics
    if METRICS_SUMMARY_PATH.exists():
        with open(METRICS_SUMMARY_PATH, "r") as f:
            benchmarks = json.load(f)

        for model_name, metrics in benchmarks.items():
            clean_cm = sanitize_for_json(metrics.get("confusion_matrix", []))
            clean_cr = sanitize_for_json(metrics.get("classification_report", {}))
            clean_hp = sanitize_for_json(metrics.get("hyperparameters", {}))

            obj, created = ModelPerformance.objects.update_or_create(
                model_name=model_name,
                defaults={
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics.get("roc_auc"),
                    "cv_mean_accuracy": metrics.get("cv_mean_accuracy", 0.0),
                    "confusion_matrix": clean_cm,
                    "classification_report": clean_cr,
                    "is_active_model": metrics.get("is_best", False),
                    "hyperparameters": clean_hp,
                },
            )
            status_str = "Created" if created else "Updated"
            print(f" -> {status_str} Model benchmark: {obj}")

    # 3. Seed Default Baseline Dataset record if raw dataset exists
    if RAW_DATA_PATH.exists() and not Dataset.objects.filter(dataset_name="Kaggle Baseline Loan Dataset").exists():
        raw_df = load_dataset(RAW_DATA_PATH)
        overview = sanitize_for_json(get_dataset_overview(raw_df))

        Dataset.objects.create(
            user=admin,
            dataset_name="Kaggle Baseline Loan Dataset",
            file="uploaded_datasets/loan_data.csv",
            file_type="CSV",
            total_records=overview["total_records"],
            total_columns=overview["total_columns"],
            missing_values_count=overview["total_missing_cells"],
            status="PROCESSED",
            summary_metadata=overview,
        )
        print(" -> Registered default Kaggle dataset in Dataset registry.")

    print("\nDATABASE SEEDING COMPLETE!")


if __name__ == "__main__":
    seed()
