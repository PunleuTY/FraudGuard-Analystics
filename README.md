# FraudGuard Credit Card Fraud Detection System

## 1. Project Overview

Fraudulent credit card transactions cause significant financial loss and reduce customer trust. Manual fraud review processes are slow, inconsistent, and not scalable. This project aims to develop a machine learning-based system to automatically detect suspicious credit card transactions using historical transaction data.

The system performs exploratory data analysis (EDA), builds multiple machine learning models, evaluates performance using appropriate metrics for imbalanced classification, and deploys the final model through a user-friendly Streamlit web interface.

---

## 2. Problem Statement

Fraud detection is a classification problem where the objective is to identify fraudulent transactions among legitimate ones. The dataset is highly imbalanced, meaning fraudulent cases represent a small portion of total transactions.

Traditional accuracy metrics are not sufficient in this scenario. Therefore, the model evaluation focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall Curve

The system aims to reduce false negatives (missed fraud) while minimizing false positives (incorrectly flagged legitimate transactions).

---

## 3. Objectives

- Perform data cleaning and exploratory data analysis
- Identify meaningful patterns and insights
- Build multiple machine learning models
- Handle class imbalance effectively
- Evaluate and compare model performance
- Optimize classification threshold
- Deploy a simple and functional web interface
- Document the full development process

---

## 4. Dataset

Source: Kaggle  
Dataset Link: https://www.kaggle.com/datasets/kartik2112/fraud-detection

The dataset contains historical credit card transaction records labeled as fraudulent or legitimate.

---

## 5. Project Scope (4 Weeks)

### In Scope

- Problem Definition
- Dataset Preparation
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Development
- Model Evaluation
- Threshold Optimization
- Streamlit Web Interface
- Documentation and Reporting

### Out of Scope (Limitations)

- Real-time streaming fraud detection
- Production-grade security implementation
- Advanced deep learning architectures
- Multi-dataset generalization studies
- Concept drift handling

---

## 6. Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── models/
│
├── reports/
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

- `data/` → Raw and processed datasets
- `notebooks/` → EDA and experimentation
- `src/` → Modularized pipeline and reusable code
- `models/` → Saved trained models
- `reports/` → Figures and final report
- `app/` → Streamlit deployment

---

## 7. Technology Stack

### Programming & Analysis

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE experiments)
- SHAP (model explainability)

### Web Interface

- Streamlit

---

## 8. Methodology

### Step 1: Data Exploration

- Class distribution analysis
- Transaction amount comparison
- Correlation analysis
- Fraud pattern investigation

### Step 2: Data Preprocessing

- Missing value handling
- Encoding categorical variables
- Feature scaling
- Stratified train-test split

### Step 3: Modeling

Baseline and advanced models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (XGBoost)

### Step 4: Handling Imbalance

- Class weighting
- SMOTE experimentation
- Threshold adjustment

### Step 5: Model Evaluation

Metrics used:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Threshold tuning is performed to balance fraud detection performance and false alarm rate.

---

## 9. Final Model Selection

The final model is selected based on:

- Highest Recall with acceptable Precision
- Stable ROC-AUC performance
- Generalization ability on test data
- Interpretability and feature importance analysis

---

## 10. Streamlit Application

The deployed application allows users to:

- Input transaction features
- View fraud probability prediction
- Receive classification result (Fraud / Legitimate)
- Observe model confidence

To run the application:

```bash
streamlit run app/streamlit_app.py
```

---

## 11. Installation

- Clone the repository

```bash
git clone https://github.com/PunleuTY/FraudGuard-Analystics
```

- Create and activate Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. Key Learnings

This project demonstrates:

- Handling imbalanced datasets
- Cost-sensitive evaluation strategies
- Threshold optimization techniques
- Model comparison methodology
- Deployment of ML models into a web interface

---

## 13. Future Improvements

- Real-time fraud detection pipeline
- Concept drift monitoring
- Ensemble stacking models
- Advanced anomaly detection techniques
- Deployment to cloud infrastructure
- Integration with transaction databases

---

## 14. Conclusion

This project presents a structured machine learning pipeline for detecting fraudulent credit card transactions. While simplified for academic purposes, it follows industry-relevant practices such as proper evaluation metrics, imbalance handling, model comparison, and deployment.

The system demonstrates how machine learning can assist in improving fraud detection efficiency and reducing financial risk.
