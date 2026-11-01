<h1 align="center">Loan Approval Prediction System</h1>
<h3 align="center">Using Data Mining, Machine Learning and Django Web Application</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/Django-6.1-darkgreen?logo=django" alt="Django" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.9-orange?logo=scikit-learn" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Frontend-Bootstrap%205%20%7C%20Chart.js-purple?logo=bootstrap" alt="Bootstrap 5" />
  <img src="https://img.shields.io/badge/Model%20Accuracy-85.37%25-brightgreen" alt="Accuracy" />
  <img src="https://img.shields.io/badge/Deployment-Render%20Cloud-blueviolet?logo=render" alt="Render" />
</p>

---

## 🌐 Live Production Deployment

- **Public Live Application:** [https://loan-approval-prediction-system-using-vpq8.onrender.com/](https://loan-approval-prediction-system-using-vpq8.onrender.com/)
- **Default Academic Demo Credentials:**
  - **Username:** `admin`
  - **Password:** `admin123`

---

## 📑 Project Abstract & Objective

Financial credit risk assessment is a central operation in commercial and retail banking. Traditional underwriting workflows often encounter bottlenecks, prolonged turnaround times, and subjective human discretion. 

This project delivers an end-to-end, academic-grade **Loan Approval Prediction System** developed using **Data Mining**, **Machine Learning**, and a modular **Django Web Framework**. The system analyzes applicant demographic and financial attributes, executes statistical data cleaning (median/mode imputation, deduplication, and Interquartile Range Winsorization for outliers), and encapsulates preprocessors in Scikit-Learn pipelines to prevent data leakage. Three candidate classifiers—**Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**—are benchmarked via Stratified 5-Fold Cross Validation. The champion model (**Logistic Regression**, achieving **85.37% test accuracy** and **90.32% F1-score**) is serialized using Joblib and deployed in an interactive financial analytics web portal.

> **Important Educational Disclaimer:** This system is an educational decision-support prediction platform. It does not execute legal or binding financial credit approval decisions.

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose & Rationale |
| :--- | :--- | :--- | :--- |
| **Programming Language** | Python | 3.11 / 3.12 / 3.13 | Primary programming environment for data mining and backend. |
| **Data Mining** | Pandas, NumPy | >= 2.2.0, >= 2.1.0 | Vectorized DataFrame transformation, matrix operations, null imputation. |
| **Statistical Visualizations** | Matplotlib, Seaborn | >= 3.9.0, >= 0.13.0 | Static high-DPI rendering for academic thesis reports & analytics dashboard. |
| **Client Analytics** | Chart.js | 4.4.x (CDN) | Interactive, responsive doughnut, bar, and pie charts in browser. |
| **Machine Learning** | Scikit-Learn | >= 1.5.0 | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, ML classifiers. |
| **Model Persistence** | Joblib | >= 1.4.0 | High-performance serialization of fitted pipelines (`.joblib`). |
| **Web Framework** | Django | 6.1.x / 5.1.x | Batteries-included MVC/MVT architecture, ORM, CSRF-secured forms. |
| **Database** | SQLite3 | Native | ACID-compliant relational storage for datasets, metrics, and audit logs. |
| **Frontend UI** | HTML5, CSS3, Bootstrap 5 | 5.3.x | Modern corporate financial theme (`#0F172A`, `#1E40AF`, `#0284C7`). |
| **Production Server** | Gunicorn & WhiteNoise | 26.x, 6.12.x | WSGI production server with compressed static asset streaming. |

---

## 📊 Dataset Specifications

The baseline dataset is the official **Kaggle Loan Prediction Problem Dataset** containing 614 records across 13 columns:

| Column Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `Loan_ID` | String | Unique applicant record identifier | e.g. `LP001002` |
| `Gender` | Categorical | Applicant gender | `Male`, `Female` |
| `Married` | Categorical | Marital status | `Yes`, `No` |
| `Dependents` | Categorical | Number of family dependents | `0`, `1`, `2`, `3+` |
| `Education` | Categorical | Educational qualification | `Graduate`, `Not Graduate` |
| `Self_Employed` | Categorical | Self-employment status | `Yes`, `No` |
| `ApplicantIncome` | Numeric | Monthly applicant income | `$150` – `$81,000` |
| `CoapplicantIncome`| Numeric | Monthly co-applicant income | `$0` – `$41,667` |
| `LoanAmount` | Numeric | Requested loan volume in thousands | `$9` – `$700` ($000s) |
| `Loan_Amount_Term`| Numeric | Duration of credit in months | `12` – `480` months |
| `Credit_History` | Numeric | Compliance with credit credit guidelines | `1.0` (Meets), `0.0` (Poor) |
| `Property_Area` | Categorical | Applicant residential location | `Urban`, `Semiurban`, `Rural` |
| **`Loan_Status`** | **Target** | **Approval classification outcome** | **`Y` (Approved), `N` (Rejected)** |

---

## 🏗️ Complete Directory Structure

```text
Loan-Approval-Prediction-System/
│
├── data/                               # Dataset Storage
│   ├── raw/
│   │   └── loan_data.csv               # Baseline Kaggle Loan Dataset (614 rows x 13 cols)
│   └── processed/                      # Preprocessed train/test partitions
│
├── src/                                # Core Data Mining & ML Package
│   ├── __init__.py
│   ├── config.py                       # Project paths, feature schemas, hyperparameters
│   ├── data_loader.py                  # Ingestion for CSV & XLSX, validation, summary statistics
│   ├── data_cleaning.py                # Null handling, deduplication, categorical standardization
│   ├── outlier_detection.py            # Interquartile Range (IQR) detection & Winsorization capping
│   ├── eda.py                          # High-DPI Seaborn/Matplotlib visualization generator
│   ├── preprocessing.py                # Scikit-learn Pipeline & ColumnTransformer builder
│   ├── feature_selection.py            # Tree-based Gini feature importance calculation
│   ├── model_training.py               # Stratified 5-Fold CV & classifier benchmarking
│   ├── model_evaluation.py             # Accuracy, Precision, Recall, F1, Confusion Matrices
│   ├── prediction.py                   # Production inference wrapper with pipeline caching
│   ├── test_phase1.py                  # Ingestion & cleaning test script
│   └── train_pipeline.py               # Complete end-to-end ML training execution runner
│
├── models/                             # Persisted ML Artifacts
│   ├── best_loan_pipeline.joblib       # Serialized champion pipeline (imputer + scaler + model)
│   └── model_metadata.json             # Model training metadata & test scores
│
├── visualizations/                     # High-DPI Publication Figures (PNGs)
│   ├── loan_status_distribution.png    # Target class balance
│   ├── applicant_income_distribution.png # Income histograms with KDE curves
│   ├── loan_amount_distribution.png    # Loan volume and terms distribution
│   ├── credit_history_vs_status.png    # Credit compliance correlation analysis
│   ├── correlation_heatmap.png         # Pearson correlation matrix
│   ├── outliers_boxplots.png           # Numerical quartile spreads & outlier markers
│   ├── feature_importance.png          # Random Forest Gini feature importances
│   └── confusion_matrix_logistic_regression.png # Champion model confusion matrix
│
├── reports/                            # Academic Documentation
│   ├── academic_project_report.md      # Comprehensive 18-section university submission report
│   └── model_comparison_summary.json  # Benchmark metrics JSON export
│
├── django_app/                         # Django Web Application Project
│   ├── manage.py                       # Django CLI entrypoint
│   ├── config/                         # Project settings, URL routing, WSGI/ASGI
│   │   ├── settings.py                 # Whitenoise, security middleware, installed apps
│   │   ├── urls.py                     # Root URL dispatcher
│   │   └── wsgi.py
│   ├── accounts/                       # User authentication (Login, Register, Logout)
│   ├── dashboard/                      # KPI aggregation & overview dashboard
│   ├── datasets/                       # Upload, preview table, preprocessing audit, CSV export
│   ├── analytics/                      # Interactive Chart.js feeds & static plots gallery
│   ├── predictions/                    # Real-time inference form, audit history, models overview
│   ├── static/                         # Assets: Custom financial CSS, JavaScript charts
│   ├── templates/                      # Semantic HTML5 Bootstrap 5 templates
│   ├── seed_database.py                # Database population script with trained metrics
│   └── test_system.py                  # Automated test suite (11 academic test cases)
│
├── build.sh                            # Cloud build script (collectstatic, migrate, seed)
├── Procfile                            # Process definition for Gunicorn
├── render.yaml                         # Infrastructure blueprint for Render deployment
├── requirements.txt                    # Pinned production dependencies
└── README.md                           # Master project documentation
```

---

## 🔬 Machine Learning Benchmarking & Measured Results

The candidate classifiers were trained on an **80% training partition** and evaluated on a **stratified held-out 20% test partition (123 records)**:

```text
-----------------------------------------------------------------------------------------------
Model Name               | Test Accuracy | Precision | Recall    | F1-Score  | 5-Fold CV Mean  
-----------------------------------------------------------------------------------------------
Logistic Regression *    | 85.37%        | 83.17%    | 98.82%    | 90.32%    | 79.83%          
Random Forest Classifier | 85.37%        | 83.17%    | 98.82%    | 90.32%    | 79.42%          
Decision Tree Classifier | 83.74%        | 83.51%    | 95.29%    | 89.01%    | 77.39%          
-----------------------------------------------------------------------------------------------
* Selected Active Champion: Logistic Regression (highest composite balance of F1-Score and CV stability)
```

### Confusion Matrix (Logistic Regression on Test Set)
- **True Negatives (TN):** `20` (Correctly predicted loan rejections)
- **False Positives (FP):** `18` (High-risk applicants misclassified as approved)
- **False Negatives (FN):** `1` (Extremely low credit rejection error rate)
- **True Positives (TP):** `84` (Correctly predicted loan approvals)

### Feature Importance Ranking (Gini Metric)
1. **`Credit_History`** (`37.0%`): The primary determinant of loan approval. Applicants meeting credit guidelines achieve a ~79% approval rate versus <10% for non-compliant applicants.
2. **`ApplicantIncome`** (`14.4%`): Monthly cash flow.
3. **`LoanAmount`** (`13.5%`): Requested credit leverage.
4. **`CoapplicantIncome`** (`8.4%`): Additional debt service coverage.
5. **`Property_Area_Semiurban`** (`3.7%`): Geographic risk factor.

---

## 🖥️ System Modules & Interface Pages

1. **Authentication (`/accounts/login/`, `/accounts/register/`)**:
   - Role-based session management, password hashing, and CSRF protection.
2. **Executive Dashboard (`/`)**:
   - Real-time KPI stat cards (Total Records, Active Model Accuracy, Missing Data Cells, Total Inferences).
   - Interactive Chart.js Doughnut chart of target distribution.
   - Recent real-time assessment audit table with status and risk badges.
3. **Dataset Ingestion & Upload (`/datasets/upload/`)**:
   - Ingest custom `.csv` and `.xlsx` files with client-side and server-side file type and 10MB size limits.
4. **Dataset Preview Table (`/datasets/preview/`)**:
   - Paginated display of dataset records with missing-cell indicators and column data types.
5. **Data Preprocessing Audit (`/datasets/preprocessing/`)**:
   - Before-and-after missing data comparison table.
   - Interquartile Range (IQR) outlier capping summary (Q1, Q3, Lower Bound, Upper Bound).
   - One-click button to download the preprocessed clean dataset CSV.
6. **Visual Analytics (`/analytics/`)**:
   - Interactive Chart.js widgets for Property Area, Credit History, and Income Brackets.
   - High-DPI gallery of 7 Seaborn exploratory charts for academic report generation.
7. **Model Performance (`/predict/models/`)**:
   - Side-by-side benchmark comparison of all 3 models with confusion matrices and hyperparameters.
8. **Real-Time Loan Predictor (`/predict/`)**:
   - Interactive 11-feature web form.
   - Instant decision outcome card (LOAN APPROVED / LOAN REJECTED), calibrated confidence progress bars, and risk ratings (Low Risk, Moderate Risk, High Risk).
9. **Prediction Audit Trail (`/predict/history/`)**:
   - Comprehensive searchable and filterable history of all past predictions with CSV export.

---

## ⚡ Local Installation & Execution Guide

### 1. Clone Repository
```powershell
git clone https://github.com/iamujjwal69/-Loan-Approval-Prediction-System-Using-Data-Mining-Machine-Learning-and-Django-Web-Application.git
cd -Loan-Approval-Prediction-System-Using-Data-Mining-Machine-Learning-and-Django-Web-Application
```

### 2. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Machine Learning Pipeline (Optional Retraining)
```powershell
python src/train_pipeline.py
```

### 5. Apply Database Migrations & Seed Default Data
```powershell
cd django_app
python manage.py migrate
python seed_database.py
```

### 6. Start Development Web Server
```powershell
python manage.py runserver 127.0.0.1:8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser.

---

## 🧪 Automated Academic Test Suite

The system includes an automated test runner validating all 11 core academic test cases:

```powershell
cd django_app
python test_system.py
```

**Test Verification Matrix:**
- `TC-01`: Valid CSV upload & validation — **PASSED**
- `TC-02`: Valid XLSX upload & parsing — **PASSED**
- `TC-03`: Invalid file format rejection (.exe) — **PASSED**
- `TC-04`: Missing value detection & imputation — **PASSED**
- `TC-05`: Duplicate detection and removal — **PASSED**
- `TC-06`: Outlier detection via IQR method & Winsorization — **PASSED**
- `TC-07`: Model training & Scikit-Learn pipeline persistence — **PASSED**
- `TC-08`: Real-time prediction with valid applicant features — **PASSED**
- `TC-09`: Prediction error handling on invalid/missing input — **PASSED**
- `TC-10`: Unauthorized dashboard access redirect guard (HTTP 302) — **PASSED**
- `TC-11`: Relational audit logging of predictions in SQLite — **PASSED**

---

## ☁️ Production Deployment on Render

This project contains native configuration files for deployment to **Render.com**:

1. **`build.sh`**: Runs `pip install`, `collectstatic`, `migrate`, and `seed_database.py`.
2. **`Procfile`**: Launches `gunicorn --chdir django_app config.wsgi:application --bind 0.0.0.0:$PORT`.
3. **`render.yaml`**: Infrastructure blueprint specifying Python runtime and environment settings.

To deploy your own instance on Render:
1. Fork or push this repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/), click **New +** $\rightarrow$ **Web Service**.
3. Connect your repository.
4. Set Build Command to `./build.sh` and Start Command to `gunicorn --chdir django_app config.wsgi:application --bind 0.0.0.0:$PORT`.
5. Select the **Free** plan and click **Deploy**.

---

## 👨‍🎓 Academic Submission Details

- **Student / Author:** Ujjwal Sharma (`iamujjwal69`)
- **Project Topic:** Loan Approval Prediction System Using Data Mining, Machine Learning and Django Web Application
- **Complete Academic Report:** [reports/academic_project_report.md](file:///d:/ary%20project/reports/academic_project_report.md)
- **Live Demo Link:** [https://loan-approval-prediction-system-using-vpq8.onrender.com/](https://loan-approval-prediction-system-using-vpq8.onrender.com/)
