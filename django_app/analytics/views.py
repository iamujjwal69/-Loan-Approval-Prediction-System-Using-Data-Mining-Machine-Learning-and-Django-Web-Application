"""
Views for Analytics App: Interactive Visual Analytics & Chart.js Data Feeds.
"""

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN
from src.data_loader import load_dataset


@login_required
def analytics_dashboard_view(request):
    df = load_dataset(RAW_DATA_PATH)

    # Static plot filenames available in /static/ (mapped to visualizations/)
    static_plots = [
        {"title": "Loan Approval Status Class Balance", "file": "loan_status_distribution.png", "desc": "Target distribution showing 422 approved (68.7%) vs 192 rejected (31.3%) applications."},
        {"title": "Applicant & Coapplicant Income Distribution", "file": "applicant_income_distribution.png", "desc": "Right-skewed income histograms with KDE curves."},
        {"title": "Loan Amount & Term Distribution", "file": "loan_amount_distribution.png", "desc": "Spread of requested credit amounts and loan duration terms in months."},
        {"title": "Credit History vs Loan Status", "file": "credit_history_vs_status.png", "desc": "Shows credit history guidelines meet rate directly correlating with ~79% approval vs 8% for non-compliant."},
        {"title": "Feature Correlation Heatmap", "file": "correlation_heatmap.png", "desc": "Multivariate Pearson correlation coefficients among continuous predictors."},
        {"title": "Numerical Feature Box Plots", "file": "outliers_boxplots.png", "desc": "Visualizes quartile spreads and flagged outliers across income and loan variables."},
        {"title": "Gini Feature Importance Ranking", "file": "feature_importance.png", "desc": "Predictive weight extracted from Random Forest ensemble."},
    ]

    return render(request, "analytics/charts.html", {"static_plots": static_plots})


@login_required
def api_chart_data(request):
    """
    Returns JSON formatted payloads for client-side Chart.js rendering.
    """
    df = load_dataset(RAW_DATA_PATH)

    # 1. Loan Status Distribution
    status_counts = df[TARGET_COLUMN].value_counts().to_dict()

    # 2. Property Area Distribution
    property_counts = df["Property_Area"].value_counts().to_dict()

    # 3. Education vs Approval
    edu_cross = pd.crosstab(df["Education"], df[TARGET_COLUMN]).to_dict(orient="index")

    # 4. Credit History Breakdown
    credit_counts = df["Credit_History"].value_counts().to_dict()
    credit_data = {
        "Meets Guidelines (1.0)": credit_counts.get(1.0, 0),
        "Non-Compliant (0.0)": credit_counts.get(0.0, 0),
    }

    # 5. Income brackets
    income_bins = [0, 2500, 5000, 7500, 10000, 25000, 100000]
    income_labels = ["<$2.5k", "$2.5k-$5k", "$5k-$7.5k", "$7.5k-$10k", "$10k-$25k", ">$25k"]
    income_binned = pd.cut(df["ApplicantIncome"], bins=income_bins, labels=income_labels).value_counts().sort_index().to_dict()

    payload = {
        "status_distribution": {
            "labels": ["Approved (Y)", "Rejected (N)"],
            "data": [status_counts.get("Y", 0), status_counts.get("N", 0)],
        },
        "property_area": {
            "labels": list(property_counts.keys()),
            "data": list(property_counts.values()),
        },
        "credit_history": {
            "labels": list(credit_data.keys()),
            "data": list(credit_data.values()),
        },
        "income_brackets": {
            "labels": list(income_binned.keys()),
            "data": list(income_binned.values()),
        },
        "education_approval": edu_cross,
    }
    return JsonResponse(payload)
