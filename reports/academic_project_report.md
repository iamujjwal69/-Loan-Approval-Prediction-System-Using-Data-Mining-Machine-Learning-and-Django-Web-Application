# Academic Project Report: Loan Approval Prediction System Using Data Mining, Machine Learning & Django Web Application

**Course / Degree:** Bachelor of Technology / Computer Science & Engineering (Data Science)  
**Project Title:** Loan Approval Prediction System Using Data Mining, Machine Learning and Django Web Application  
**Version:** Final Submission 1.0  
**Technology Stack:** Python 3.13, Scikit-Learn, Pandas, NumPy, Django 6.1, Bootstrap 5, Chart.js, SQLite  

---

## 1. Project Abstract
Financial credit evaluation is a critical activity in banking and lending institutions. Traditional manual assessment often encounters bottlenecks, processing delays, and subjectivity. This project develops an end-to-end, automated **Loan Approval Prediction System** leveraging Data Mining techniques, Machine Learning classification algorithms, and a modular Django web application. The platform ingests applicant financial attributes, executes statistical data cleaning (handling nulls via median/mode imputation, removing duplicates, and applying Interquartile Range Winsorization for outliers), and encapsulates preprocessors in Scikit-Learn pipelines. Three candidate models—**Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**—are evaluated via Stratified 5-Fold Cross Validation. The champion model (Logistic Regression, achieving **85.37% test accuracy** and **90.32% F1-score**) is serialized using Joblib and integrated into an interactive financial dashboard providing real-time credit predictions, probability gauges, and risk ratings.

---

## 2. Introduction
With the exponential growth of personal and commercial borrowing, automated decision-support systems have become central to risk mitigation. Data mining enables extraction of hidden applicant patterns from historical credit records, while machine learning algorithms infer non-linear decision boundaries to forecast creditworthiness. This project demonstrates an academic implementation adhering to strict machine learning best practices: total isolation of test data to prevent data leakage, comprehensive exploratory visual analytics, and an enterprise-grade Django web application designed for loan officers and academic analysts.

---

## 3. Problem Statement
Manual loan evaluation workflows face numerous challenges:
- High operational overhead and prolonged turnaround times.
- Human bias and inconsistent application of underwriting rules across branch officers.
- Inability to quickly detect complex multivariate interactions among applicant income, requested loan volume, and credit history.
- The risk of data leakage and overfitting when data science workflows do not properly isolate preprocessing transformations.

This project resolves these issues by delivering a transparent, reproducible, and leak-free machine learning pipeline coupled with an interactive web dashboard.

---

## 4. Objectives
1. **Automate Data Ingestion & Validation**: Ingest both CSV and Excel datasets, verifying attribute schemas against expected financial parameters.
2. **Execute Rigorous Data Mining**: Quantify and impute missing values, identify and eliminate duplicates, and cap extreme outliers via Interquartile Range (IQR) Winsorization without discarding valid applicant samples.
3. **Conduct Exploratory Data Analysis (EDA)**: Produce high-DPI distribution plots, correlation heatmaps, and Gini feature importances.
4. **Train and Compare Machine Learning Classifiers**: Benchmark Logistic Regression, Decision Tree, and Random Forest models across Accuracy, Precision, Recall, F1-Score, and Cross-Validation metrics.
5. **Persist and Deploy Production Pipeline**: Serialize the winning preprocessor and estimator pipeline using Joblib for real-time inference.
6. **Deliver an Intuitive Web Dashboard**: Provide user authentication, dynamic Chart.js visualizations, real-time prediction forms, and audited decision histories with CSV export.

---

## 5. Existing System
In existing legacy banking setups:
- Loan evaluation relies on rigid rule-based scorecards or manual underwriting spreadsheets.
- Outliers and missing records are often either blindly removed (causing sample bias) or manually resolved.
- Predictive models are rarely integrated directly with self-service web dashboards, creating a gap between data science prototypes and operational deployment.

---

## 6. Proposed System
The proposed system features:
- **Modular Pipeline Design**: Preprocessing, outlier handling, training, and evaluation are separated into distinct Python modules under `src/`.
- **Leak-Free Pipelines**: Imputers, encoders, and scalers are wrapped within Scikit-Learn `ColumnTransformer` and fitted exclusively on training splits.
- **Controlled Ingestion & Inspection**: Allows users to upload custom datasets, preview paginated rows, inspect missing value statistics, and download preprocessed datasets.
- **Model Governance**: Automatically logs model performance, confusion matrices, and hyperparameters to SQLite.
- **Instant Decisioning & Auditing**: Generates real-time loan approval predictions, outputs calibrated probabilities, computes risk levels (Low, Moderate, High), and logs every evaluation to a searchable audit trail.

---

## 7. Software and Hardware Requirements

### Software Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Programming Language:** Python 3.11 / 3.12 / 3.13
- **Data Science Libraries:** Pandas 3.1+, NumPy 2.5+, Scipy 1.18+, Scikit-Learn 1.9+
- **Data Visualization:** Matplotlib 3.11+, Seaborn 0.13+, Chart.js 4.4+
- **Web Framework:** Django 6.1+
- **Database:** SQLite3
- **Model Serialization:** Joblib 1.6+
- **Development Tool:** Visual Studio Code / Terminal

### Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Quad-core recommended)
- **RAM:** Minimum 4 GB (8 GB recommended)
- **Storage:** Minimum 500 MB free disk space for repository, virtual environment, and model artifacts

---

## 8. System Architecture
The application is structured into four decoupled layers:
1. **Client Layer:** Responsive HTML5, Bootstrap 5, CSS3, and dynamic Chart.js graphics.
2. **Django Web Application Layer:** Handles URL routing, session security, CSRF protection, and view controllers (`accounts`, `dashboard`, `datasets`, `analytics`, `predictions`).
3. **Machine Learning & Data Mining Layer (`src/`):** Executes data ingestion, IQR capping, feature encoding (`ColumnTransformer`), model training, cross-validation, and production inference.
4. **Data & Persistence Layer:** SQLite relational database, Joblib binary serialized pipeline (`best_loan_pipeline.joblib`), metadata JSONs, and static visualization charts.

---

## 9. Data Mining Methodology

### A. Data Ingestion & Dataset Provenance
The system ingests the official **Kaggle Loan Prediction Problem Dataset** (Kaggle dataset URL: `https://www.kaggle.com/datasets/ninzaami/loan-predication`), mirrored via the canonical research repository `https://raw.githubusercontent.com/metis-macys-66898/data622_fa2021/main/hw3/data/Loan_approval.csv` and stored locally at `data/raw/loan_data.csv`. The dataset comprises 614 applicant records across 13 columns. The schema validation engine ensures the presence of the 11 predictive features and the binary target variable `Loan_Status`.

### B. Missing Value Treatment
Missing data is categorized by data type and imputed:
- **Numerical Features** (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`): Imputed using the **Median** to protect against skewness.
- **Categorical Features** (`Gender`, `Married`, `Dependents`, `Self_Employed`): Imputed using the **Mode (most frequent value)**.

### C. Outlier Detection & IQR Winsorization
The Interquartile Range (IQR) method is applied to continuous variables:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = \max(0, Q_1 - 1.5 \times \text{IQR}), \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
Extreme values exceeding these bounds are capped (Winsorized) to preserve valid observations rather than dropping valuable records.

### D. Feature Transformation
- **Categorical Variables:** Encoded using Scikit-Learn `OneHotEncoder(handle_unknown='ignore')`.
- **Numerical Variables:** Standardized using `StandardScaler` to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).

---

## 10. Algorithm Explanation

### 1. Logistic Regression
A linear classification model that models the posterior probability of loan approval ($y=1$) using the logistic sigmoid function:
$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
Optimized using the `liblinear` coordinate descent solver with $L_2$ regularization.

### 2. Decision Tree Classifier
A non-parametric supervised learning method that partitions the feature space into orthogonal regions by recursively maximizing information gain using the Gini Impurity criterion:
$$\text{Gini}(D) = 1 - \sum_{i=1}^C p_i^2$$
Tuned with `max_depth=4` and `min_samples_split=6` to prevent overfitting.

### 3. Random Forest Classifier
An ensemble learning technique that constructs 100 decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace selection. The final prediction aggregates tree outputs via majority voting:
$$\hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), \dots, h_B(\mathbf{x})\}$$

---

## 11. Implementation Details
The codebase is structured into modular Python packages:
- `src/config.py`: Single source of truth for constants, paths, and hyperparameters.
- `src/data_loader.py`: Handles CSV/Excel loading and validation.
- `src/data_cleaning.py`: Performs null analysis, deduplication, and string cleaning.
- `src/outlier_detection.py`: Computes IQR boundaries and applies capping.
- `src/eda.py`: Renders high-DPI figures saved to `visualizations/`.
- `src/preprocessing.py`: Builds the Scikit-Learn `ColumnTransformer`.
- `src/model_training.py`: Trains all models with StratifiedKFold cross-validation.
- `src/model_evaluation.py`: Generates confusion matrices and classification reports.
- `src/prediction.py`: Implements cached pipeline inference for web requests.

---

## 12. List of Generated Figures & Visualizations
The system automatically generates and renders the following figures in `visualizations/`:
1. `loan_status_distribution.png`: Target class balance (422 Approved vs 192 Rejected).
2. `applicant_income_distribution.png`: Income distribution and KDE curves.
3. `loan_amount_distribution.png`: Requested loan volume histogram and term distribution.
4. `credit_history_vs_status.png`: Stacked bar chart showing approval percentage by credit compliance.
5. `correlation_heatmap.png`: Pearson correlation matrix among numerical features.
6. `outliers_boxplots.png`: Box plots for ApplicantIncome, CoapplicantIncome, LoanAmount, and Term.
7. `feature_importance.png`: Gini importance ranking from Random Forest.
8. `confusion_matrix_logistic_regression.png`: Confusion matrix for the champion classifier.

---

## 13. Results and Discussion

### Model Benchmark Summary (Evaluated on Held-out 20% Test Split: 123 records)

| Model Name | Test Accuracy | Precision | Recall | F1-Score | 5-Fold CV Mean | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **85.37%** | **83.17%** | **98.82%** | **90.32%** | **79.83%** | **Selected Champion** |
| **Random Forest** | 85.37% | 83.17% | 98.82% | 90.32% | 79.42% | Candidate |
| **Decision Tree** | 83.74% | 83.51% | 95.29% | 89.01% | 77.39% | Candidate |

### Confusion Matrix Analysis (Logistic Regression)
- **True Negatives (TN):** 20 (Correctly predicted loan rejections)
- **False Positives (FP):** 18 (Rejected applicants predicted as approved)
- **False Negatives (FN):** 1 (Approved applicants predicted as rejected)
- **True Positives (TP):** 84 (Correctly predicted loan approvals)

### Key Analytical Insights
1. **Credit History is the Dominant Predictor:** Random Forest Gini importance scores assign **37.0%** of total predictive weight to `Credit_History`. Applicants meeting credit guidelines achieve a ~79% approval rate, compared to under 10% for non-compliant applicants.
2. **Financial Scale:** Applicant Income (**14.4%**) and Loan Amount (**13.5%**) are the second and third most influential features.
3. **Generalization:** Stratified 5-Fold cross-validation confirmed consistent generalization accuracy (CV Mean ~79.8%) without evidence of variance overfitting.

---

## 14. Testing & Test Cases Matrix

All 11 required system test cases were implemented in `django_app/test_system.py` and passed with 100% success:

| Test ID | Test Case Description | Expected Result | Actual Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **TC-01** | Valid CSV upload | Accepted, schema validated, stored in database | Ingested and registered with status 200 | **PASS** |
| **TC-02** | Valid XLSX upload | Excel format parsed, overview extracted | Ingested and registered with status 200 | **PASS** |
| **TC-03** | Invalid file upload (.exe) | Blocked by extension/size validator | Rejected with form validation error | **PASS** |
| **TC-04** | Missing value handling | Imputes numerical (median) & categorical (mode) | All 7 columns resolved, 0 remaining nulls | **PASS** |
| **TC-05** | Duplicate removal | Identifies and strips identical records | Exactly 10 duplicates removed | **PASS** |
| **TC-06** | Outlier detection | Identifies IQR boundaries and caps values | Bounds computed, 50 outliers capped | **PASS** |
| **TC-07** | Model training & persistence | Serializes pipeline to disk via Joblib | Artifact verified on disk | **PASS** |
| **TC-08** | Prediction with valid input | Generates class, probability, and risk rating | Class 'Y', 91.9% approval prob, Low Risk | **PASS** |
| **TC-09** | Prediction with invalid input | Catches missing mandatory feature | `ValueError` properly raised and caught | **PASS** |
| **TC-10** | Unauthorized access guard | Redirects unauthenticated user to login | HTTP 302 Redirect to `/accounts/login/` | **PASS** |
| **TC-11** | Prediction history storage | Audits inference with user and timestamp | Saved to SQLite, verified by query | **PASS** |

---

## 15. Limitations
1. **Educational Scope:** Designed as an educational decision-support tool; does not incorporate real-time external credit bureau APIs (e.g., CIBIL/Experian).
2. **Tabular Scope:** The dataset is tabular; unstructured credit documentation (such as PDF bank statements or salary slips) is not parsed.
3. **Dataset Volume:** The baseline Kaggle dataset contains 614 rows, which provides effective training for linear and tree-based models, but would benefit from larger multi-institution data for deep neural models.

---

## 16. Future Scope
1. **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) and LIME to generate individual applicant waterfall plots explaining exact feature contributions.
2. **Automated Retraining Worker:** Implement Celery and Redis to handle periodic offline pipeline retraining on newly collected loan records.
3. **Integration with Credit APIs:** Add secure mock adapters for Open Banking APIs and identity verification services.
4. **Exportable PDF Reports:** Provide one-click PDF generation for individual loan assessment certificates.

---

## 17. Conclusion
The **Loan Approval Prediction System** successfully integrates modern Data Mining practices, robust Scikit-Learn Machine Learning pipelines, and a feature-rich Django web application. By enforcing leak-free cross-validation, IQR Winsorization, automated schema validation, and real-time inference caching, the system delivers an accurate (**85.37% test accuracy**), transparent, and professional platform suitable for university academic demonstration and financial analytics education.

---

## 18. References
1. Kaggle: *Loan Prediction Problem Dataset* by Ninzaami (2020).
2. Pedregosa, F., et al.: *Scikit-learn: Machine Learning in Python*, Journal of Machine Learning Research, 12, 2825-2830 (2011).
3. McKinney, W.: *Data Structures for Statistical Computing in Python*, Proceedings of the 9th Python in Science Conference (2010).
4. Django Software Foundation: *The Django Web Framework Documentation*, Version 6.1 (2026).
5. Breiman, L.: *Random Forests*, Machine Learning, 45(1), 5-32 (2001).
6. Han, J., Kamber, M., Pei, J.: *Data Mining: Concepts and Techniques*, Morgan Kaufmann (3rd Edition).
