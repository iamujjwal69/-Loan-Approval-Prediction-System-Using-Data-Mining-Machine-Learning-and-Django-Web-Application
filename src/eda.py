"""
Exploratory Data Analysis (EDA) & Visualization Generator Module.
Generates publication-quality, high-DPI charts and saves them to the visualizations directory
for integration into academic project reports and the Django analytics dashboard.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    VISUALIZATIONS_DIR,
)

# Visual styling setup
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update(
    {
        "figure.autolayout": True,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 300,
    }
)


def generate_loan_status_distribution(df: pd.DataFrame) -> Path:
    """Plot distribution of Target Variable (Loan_Status)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[TARGET_COLUMN].value_counts()
    colors = ["#22c55e" if idx == "Y" else "#ef4444" for idx in counts.index]
    
    bars = ax.bar(["Approved (Y)", "Rejected (N)"], counts.values, color=colors, width=0.5, edgecolor="#1e293b", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 5, f"{int(yval)} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    ax.set_title("Loan Approval Status Distribution (Class Balance)", fontweight="bold", pad=15)
    ax.set_ylabel("Number of Applicants")
    ax.set_ylim(0, max(counts.values) * 1.15)
    
    output_path = VISUALIZATIONS_DIR / "loan_status_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_income_distribution(df: pd.DataFrame) -> Path:
    """Plot distribution and KDE of Applicant Income and Coapplicant Income."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Applicant Income
    sns.histplot(df["ApplicantIncome"].dropna(), kde=True, ax=axes[0], color="#0284c7", bins=30, edgecolor="white")
    axes[0].set_title("Applicant Income Distribution", fontweight="bold")
    axes[0].set_xlabel("Applicant Income ($)")
    axes[0].set_ylabel("Frequency")
    
    # Coapplicant Income
    sns.histplot(df["CoapplicantIncome"].dropna(), kde=True, ax=axes[1], color="#0ea5e9", bins=30, edgecolor="white")
    axes[1].set_title("Coapplicant Income Distribution", fontweight="bold")
    axes[1].set_xlabel("Coapplicant Income ($)")
    axes[1].set_ylabel("Frequency")

    output_path = VISUALIZATIONS_DIR / "applicant_income_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_loan_amount_distribution(df: pd.DataFrame) -> Path:
    """Plot histogram & KDE of Loan Amount and Term."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["LoanAmount"].dropna(), kde=True, ax=axes[0], color="#6366f1", bins=30, edgecolor="white")
    axes[0].set_title("Loan Amount Distribution (in Thousands)", fontweight="bold")
    axes[0].set_xlabel("Loan Amount ($000s)")
    axes[0].set_ylabel("Count")

    term_counts = df["Loan_Amount_Term"].dropna().value_counts().sort_index()
    axes[1].bar(term_counts.index.astype(str), term_counts.values, color="#8b5cf6", edgecolor="#312e81")
    axes[1].set_title("Loan Term Distribution (Months)", fontweight="bold")
    axes[1].set_xlabel("Term (Months)")
    axes[1].set_ylabel("Applicants")
    axes[1].tick_params(axis="x", rotation=45)

    output_path = VISUALIZATIONS_DIR / "loan_amount_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_credit_history_analysis(df: pd.DataFrame) -> Path:
    """Analyze Loan Status based on Credit History."""
    fig, ax = plt.subplots(figsize=(7, 5))

    cross_tab = pd.crosstab(df["Credit_History"], df[TARGET_COLUMN], normalize="index") * 100
    cross_tab.plot(kind="bar", stacked=True, color=["#ef4444", "#22c55e"], ax=ax, edgecolor="#1e293b")

    ax.set_title("Loan Approval Rate by Credit History Status", fontweight="bold", pad=15)
    ax.set_xlabel("Credit History (0.0 = Poor / Guidelines Not Met, 1.0 = Meets Guidelines)")
    ax.set_ylabel("Approval Percentage (%)")
    ax.set_xticklabels(["0.0 (Poor History)", "1.0 (Good History)"], rotation=0)
    ax.legend(["Rejected (N)", "Approved (Y)"], title="Loan Status", frameon=True)
    ax.set_ylim(0, 110)

    # Annotate bars
    for n, c in enumerate([cross_tab.index]):
        for idx in range(len(cross_tab)):
            h_rej = cross_tab.iloc[idx, 0]
            h_app = cross_tab.iloc[idx, 1]
            ax.text(idx, h_rej / 2, f"{h_rej:.1f}%", ha="center", va="center", color="white", fontweight="bold")
            ax.text(idx, h_rej + (h_app / 2), f"{h_app:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    output_path = VISUALIZATIONS_DIR / "credit_history_vs_status.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Compute and plot correlation matrix for numerical features."""
    fig, ax = plt.subplots(figsize=(8, 6))

    num_df = df[NUMERICAL_FEATURES].copy()
    if TARGET_COLUMN in df.columns:
        num_df["Target_Approved"] = (df[TARGET_COLUMN] == "Y").astype(int)

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        linewidths=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=15)

    output_path = VISUALIZATIONS_DIR / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_outliers_boxplots(df: pd.DataFrame) -> Path:
    """Generate Box Plots for numerical attributes to illustrate outliers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    features_to_plot = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    colors = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6"]

    for i, col in enumerate(features_to_plot):
        if col in df.columns:
            sns.boxplot(y=df[col].dropna(), ax=axes[i], color=colors[i], flierprops={"marker": "o", "color": "#dc2626", "alpha": 0.6})
            axes[i].set_title(f"Box Plot: {col}", fontweight="bold")
            axes[i].set_ylabel(col)

    output_path = VISUALIZATIONS_DIR / "outliers_boxplots.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_all_visualizations(df: pd.DataFrame) -> dict:
    """Generates all EDA visualizations and returns dict of paths."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "loan_status": str(generate_loan_status_distribution(df)),
        "income": str(generate_income_distribution(df)),
        "loan_amount": str(generate_loan_amount_distribution(df)),
        "credit_history": str(generate_credit_history_analysis(df)),
        "correlation": str(generate_correlation_heatmap(df)),
        "outliers": str(generate_outliers_boxplots(df)),
    }
    return paths
