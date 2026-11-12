# Diabetes Risk Prediction using Logistic Regression

A foundational machine learning pipeline designed to predict diabetes onset risk using diagnostic health measurements and patient clinical indicators. This project implements exploratory data analysis, feature scaling, and L2-regularized logistic regression to classify diabetes risk and evaluate clinical trade-offs.

## Project Overview

Early identification of individuals at high risk for diabetes enables timely lifestyle interventions and preventive clinical care. This project processes diagnostic patient records and trains a penalized logistic regression classifier to predict binary diabetes outcomes (`0` = Non-Diabetic, `1` = Diabetic).

### Key Highlights

* **Dataset:** Public [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) comprising 768 patient records across 8 diagnostic features (pregnancies, plasma glucose concentration, diastolic blood pressure, triceps skinfold thickness, 2-hour serum insulin, BMI, diabetes pedigree function, and age).
* **Exploratory Data Analysis (EDA):** Inspected feature distributions, verified absence of missing values and duplicates, and generated distribution plots and correlation matrices to analyze clinical predictors such as glucose levels and BMI.
* **Feature Scaling:** Applied `StandardScaler` to normalize continuous clinical metrics prior to model training to ensure balanced gradient updates.
* **Modeling:** Trained a `LogisticRegression` model utilizing `l2` regularization and an optimized regularization parameter (`C=0.2`) to prevent overfitting.

---

## Performance Metrics

Evaluation on the held-out test split yielded solid predictive baseline performance:

* **ROC-AUC Score:** `0.74`
* **Confusion Matrix:**
  $$\begin{bmatrix} 80 & 19 \\ 18 & 37 \end{bmatrix}$$
  *(True Negatives: 80 | False Positives: 19 | False Negatives: 18 | True Positives: 37)*

* **Clinical Cost Trade-off:** In diabetes screening, **false negatives (18 cases)** carry higher clinical risk than false positives (19 cases), as missed diagnoses delay preventive care and increase long-term complication risks, whereas false positives primarily lead to low-cost confirmatory blood tests (such as HbA1c or fasting glucose).

---

## Project Structure

```text
├── CaseStudy1.ipynb         # Complete end-to-end Jupyter Notebook pipeline
└── README.md                # Project documentation
```

---

## Getting Started & Replication

You can launch and run `CaseStudy1.ipynb` directly in Google Colab or locally via Jupyter Notebook to reproduce the preprocessing, model training, and ROC-AUC evaluation workflow.
