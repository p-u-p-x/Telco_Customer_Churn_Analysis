# 📞 Telco Customer Churn: Feature Engineering, Model Building & Scaling

This project implements a full **classification pipeline** for predicting customer churn using the IBM Telco Customer Churn dataset. It covers data cleaning, feature engineering, encoding, scaling, model training, evaluation, and a research based justification of the best performing algorithm.

## 🏗️ Pipeline Overview

### High-Level Flow

**Raw CSV → Cleaning → Feature Engineering → Encoding → Scaling → Model Training → Evaluation → Algorithm Justification**

## 📂 Repository Structure

```text
telco-churn-project/
│
├── Telco_Customer_Churn_Analysis.ipynb # Full analysis notebook (Colab ready)
├── report/
│ └── Telco_Churn_Report.pdf # Short written report (2 to 4 pages)
├── README.md # Project overview and instructions
└── LICENSE # License information for the repository
```
---

## 📊 Dataset

**Source:** [Telco Customer Churn (IBM Sample Dataset)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) on Kaggle

7,043 customer records across 21 columns covering demographics, account information, subscribed services, and churn status.

| Feature Group | Columns |
|---|---|
| Identifiers (dropped) | customerID |
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account Info | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services Subscribed | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Target Variable | Churn (Yes / No) |

---

## 🧹 Task 1: Data Preparation

- **Blank string fix:** `TotalCharges` contained 11 blank strings, all belonging to customers with `tenure = 0`. Converted to numeric and filled with 0 to reflect the business reality of a brand new, unbilled customer.
- **Dropped `customerID`** since it carries no predictive signal.
- **Target check:** confirmed only `Yes` / `No` values in `Churn`, encoded to 1 / 0.
- **Class balance:** ~73.5% No Churn vs ~26.5% Churn, a mild imbalance that motivates evaluating beyond accuracy.

## 🧪 Task 2: Feature Engineering

Three engineered features, each with a stated rationale:

| Feature | Description | Why it helps |
|---|---|---|
| `tenure_group` | Buckets customers into New, Established, Long_term | Captures the nonlinear drop in churn risk as tenure increases |
| `total_services` | Count of subscribed add on / connectivity services | Higher service adoption signals customer investment and lower churn |
| `monthly_to_tenure_ratio` | `MonthlyCharges / (tenure + 1)` | Flags customers whose spend is concentrated early, a pattern linked to price sensitive churn |

**Encoding:**
- `Contract` → ordinal encoding (Month-to-month < One year < Two year), since a genuine order exists.
- All other categorical columns → one hot encoding, since no natural order exists.

## ⚖️ Task 3: Feature Scaling

- **Scale sensitive features:** `tenure`, `MonthlyCharges`, `TotalCharges`, `total_services`, `monthly_to_tenure_ratio`.
- **Scale invariant models:** Decision Tree, Random Forest, Gradient Boosting, XGBoost, Naive Bayes.
- Applied both `StandardScaler` and `MinMaxScaler`, **fit only on the training set** to avoid data leakage into the test set.
- `StandardScaler` carried forward into modeling, chosen for its robustness to the right skewed billing columns.

## 🤖 Task 4: Model Building

Split: 80/20 train test, stratified on `Churn`.

Eight classifiers trained across five families, evaluated with 5 fold stratified cross validation:

| Family | Models |
|---|---|
| Linear / Baseline | Logistic Regression |
| Tree Based | Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| Distance Based | K Nearest Neighbors |
| Margin Based | Support Vector Machine |
| Probabilistic | Naive Bayes |

**Metrics reported:** Accuracy, Precision, Recall, F1 Score, ROC AUC, plus mean and standard deviation of cross validated ROC AUC.

### 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8034 | 0.6644 | 0.5241 | 0.5859 | 0.8461 |
| Gradient Boosting | 0.7977 | 0.6551 | 0.5027 | 0.5688 | 0.8437 |
| K Nearest Neighbors | 0.7878 | 0.6119 | 0.5481 | 0.5783 | 0.8277 |
| Random Forest | 0.7885 | 0.6310 | 0.4893 | 0.5512 | 0.8254 |

*(Full table with all 8 models, plus ROC curves, confusion matrix, and feature importance / coefficient plots, is in the notebook.)*

## 🔬 Task 5: Algorithm Research & Justification

The notebook discusses, for every algorithm trained, its core assumptions (linearity, feature independence, sensitivity to scale and outliers) and relates those assumptions back to the actual properties of the Telco dataset: mixed feature types, correlated service columns, and moderate class imbalance.

Best algorithm identified: **Logistic Regression**, based on the empirical results above, supported by conceptual reasoning about why a well engineered linear model can match or beat tree ensembles on a dataset of this size and structure. Limitations and a scenario favoring a tree based alternative are also discussed.

Cited sources:
- Pedregosa et al., scikit-learn documentation, *Ensemble methods* — https://scikit-learn.org/stable/modules/ensemble.html
- Chen, T. and Guestrin, C., *"XGBoost: A Scalable Tree Boosting System"*, KDD 2016

---

## 🚀 Getting Started

### Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Open `Telco_Customer_Churn_Analysis.ipynb` directly from this repository, or upload it manually.
3. Run every cell top to bottom. The dataset downloads automatically via `kagglehub`, no manual upload needed.

### Run Locally

```bash
pip install kagglehub scikit-learn pandas numpy matplotlib seaborn xgboost jupyter
jupyter notebook Telco_Customer_Churn_Analysis.ipynb
```

A free Kaggle account may be required the first time `kagglehub` authenticates.

---

## 🛠️ Technical Features

- **Leakage safe scaling**: scalers fit strictly on the training set
- **Stratified splitting and cross validation** to respect class imbalance
- **Model agnostic evaluation loop** covering 8 classifiers in one pass
- **Dynamic best model selection** for confusion matrix and importance plots

---

## 📬 Contact

- ✉️ **Email:** your.email@example.com
- 💼 **LinkedIn:** [Your Name](https://www.linkedin.com/in/your-profile)
