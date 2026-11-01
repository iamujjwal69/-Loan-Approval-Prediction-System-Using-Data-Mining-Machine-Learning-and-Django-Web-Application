# Loan Approval Prediction System Using Data Mining, Machine Learning & Django Web Application

An end-to-end, academic-grade Data Mining and Machine Learning Web Application developed with Python 3.13, Django 6.1, Scikit-Learn, Pandas, NumPy, Bootstrap 5, and Chart.js.

The system analyzes financial applicant profiles, uncovers credit risk patterns through data mining and exploratory data analysis (EDA), trains and benchmarks three candidate machine learning classifiers, and provides real-time loan approval inference with calibrated probabilities and risk ratings.

---

## Key Features

1. **Dataset Management & Validation**:
   - Ingest custom CSV and Excel (`.xlsx`, `.xls`) datasets.
   - Comprehensive schema validation against required credit attributes.
   - Dynamic data preview table with pagination and missing-cell indicators.
2. **Data Mining & Preprocessing Pipeline**:
   - Automated null-value identification with median (numerical) and mode (categorical) imputation.
   - Interquartile Range (IQR) outlier detection with statistical Winsorization capping.
   - Encapsulated Scikit-Learn `ColumnTransformer` (StandardScaler + OneHotEncoder) strictly fitted on training splits to prevent data leakage.
   - Clean preprocessed dataset CSV export.
3. **Exploratory Data Analysis (EDA)**:
   - High-DPI Seaborn & Matplotlib visual analytics: Class balance, Income KDEs, Loan terms, Credit compliance, Outlier box plots, Pearson correlation heatmap, and Random Forest Gini feature importances.
   - Interactive client-side charts rendered with Chart.js.
4. **Machine Learning Model Governance**:
   - Comparative benchmarking of **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**.
   - Stratified 5-Fold Cross Validation.
   - Comprehensive evaluation: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
   - Automated selection and persistence of champion model via `joblib`.
5. **Real-Time Prediction Engine & Audit**:
   - Web application form for applicant evaluation.
   - Instant decision output (Approved vs. Rejected) with confidence gauges and risk classification.
   - Searchable, filterable audit history with one-click CSV export.

---

## System Architecture

```text
d:\ary project\
├── data/
│   ├── raw/loan_data.csv               # Baseline Kaggle Loan Prediction Dataset (614 rows x 13 cols)
│   └── processed/                      # Preprocessed partitions
├── src/                                # Core Data Mining & ML Package
│   ├── config.py                       # Project constants, schemas, hyperparameter settings
│   ├── data_loader.py                  # Ingestion, validation, summary analytics
│   ├── data_cleaning.py                # Null analysis, deduplication, string sanitization
│   ├── outlier_detection.py            # IQR calculation and Winsorization capping
│   ├── eda.py                          # High-DPI visualization generator
│   ├── preprocessing.py                # ColumnTransformer & Pipeline builder
│   ├── feature_selection.py            # Gini importance and correlation ranking
│   ├── model_training.py               # 5-fold CV & model benchmarking
│   ├── model_evaluation.py             # Evaluation metrics and confusion matrices
│   ├── prediction.py                   # Production inference module with pipeline caching
│   ├── test_phase1.py                  # Phase 1 unit test runner
│   └── train_pipeline.py               # End-to-end ML training execution runner
├── models/
│   ├── best_loan_pipeline.joblib       # Serialized champion pipeline
│   └── model_metadata.json             # Model training metadata & test metrics
├── visualizations/                     # High-DPI exported chart figures
├── reports/
│   ├── academic_project_report.md      # Complete university-level project submission report
│   └── model_comparison_summary.json  # Benchmarking metrics
└── django_app/                         # Django Web Application Root
    ├── manage.py
    ├── config/                         # Project settings & URL routing
    ├── accounts/                       # Authentication (Login, Register, Logout)
    ├── dashboard/                      # KPI aggregation & overview
    ├── datasets/                       # Upload, preview, preprocessing
    ├── analytics/                      # Interactive charts & API endpoints
    ├── predictions/                    # Real-time inference & history audit
    ├── static/                         # Financial CSS & JavaScript assets
    ├── templates/                      # Bootstrap 5 responsive templates
    ├── seed_database.py                # Database population script
    └── test_system.py                  # 11 automated academic test cases
```

---

## Measured Model Benchmarks (Held-out Test Split)

| Classifier | Test Accuracy | Precision | Recall | F1-Score | 5-Fold CV Mean | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **85.37%** | **83.17%** | **98.82%** | **90.32%** | **79.83%** | **Active Champion** |
| **Random Forest** | 85.37% | 83.17% | 98.82% | 90.32% | 79.42% | Candidate |
| **Decision Tree** | 83.74% | 83.51% | 95.29% | 89.01% | 77.39% | Candidate |

*Confusion Matrix (Logistic Regression on Test Set):*
- True Negative (Rejected correctly): **20**
- False Positive (Rejected misclassified): **18**
- False Negative (Approved misclassified): **1**
- True Positive (Approved correctly): **84**

---

## Installation & Setup Instructions

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13 installed on Windows/Linux/macOS.
- Visual Studio Code or preferred editor.

### 2. Activate Virtual Environment
```powershell
# Navigate to project directory
cd "d:\ary project"

# Activate existing virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Retrain / Run Machine Learning Pipeline (Optional)
To regenerate all visualizations, retrain classifiers, and verify production inference:
```powershell
python src/train_pipeline.py
```

### 4. Run Django Migrations & Seed Database
```powershell
cd django_app
python manage.py migrate
python seed_database.py
```

### 5. Launch Development Web Server
```powershell
python manage.py runserver 127.0.0.1:8000
```
Open your browser and navigate to:
**http://127.0.0.1:8000/**

---

## Default Academic Demo Credentials

- **Username:** `admin`
- **Password:** `admin123`

*(You can also register a new analyst user via the `/accounts/register/` page).*

---

## Running Automated System Verification Tests

The project includes an automated test runner validating all 11 system test cases required for university submission:

```powershell
cd "d:\ary project\django_app"
..\venv\Scripts\python test_system.py
```

**Test Coverage:**
1. Valid CSV upload
2. Valid XLSX upload
3. Invalid file upload
4. Missing value detection & imputation
5. Duplicate record removal
6. Outlier detection & IQR capping
7. Model training & pipeline persistence
8. Prediction with valid input
9. Prediction with invalid input
10. Unauthorized dashboard access security
11. Prediction history storage in SQLite
