# Telecom Churn Prediction ML System

## 🚀 Overview

This project is an end-to-end machine learning system designed to predict customer churn in a telecom dataset. It combines robust model training, validation, and deployment-ready API serving.

The goal is to identify customers likely to churn so that businesses can take proactive retention actions.

---

## 🧠 Key Features

* LightGBM-based classification model
* Stratified K-Fold cross-validation
* Class imbalance handling using `scale_pos_weight`
* Feature engineering pipeline for behavioral insights
* Hyperparameter tuning using Optuna
* REST API using FastAPI for real-time predictions
* Dockerized for reproducible deployment

---

## 📊 Model Performance

* **Final ROC-AUC:** ~0.90
* Stable cross-validation performance across folds

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* LightGBM
* Scikit-learn
* Optuna
* FastAPI
* Docker

---

## 🏗️ Project Architecture

```text
User Input → FastAPI → Feature Engineering → Model → Prediction Output
```

---

## 🔄 Workflow

1. Data preprocessing and feature engineering
2. Stratified K-Fold validation
3. Model training using LightGBM
4. Hyperparameter tuning with Optuna
5. Model saving and metadata tracking
6. API deployment using FastAPI
7. Containerization using Docker

---

## 🧪 Example Prediction

Input:

```json
{
  "AccountWeeks": 128,
  "ContractRenewal": 1,
  "DataPlan": 1,
  "DataUsage": 2.7,
  "CustServCalls": 1,
  "DayMins": 265.1,
  "DayCalls": 110,
  "MonthlyCharge": 89.0,
  "OverageFee": 9.87,
  "RoamMins": 10.0
}
```

Output:

```json
{
  "churn_probability": 0.18,
  "churn_prediction": 0
}
```

---

## 🧩 Skills Demonstrated

* Machine learning pipeline design
* Feature engineering and validation strategy
* Handling imbalanced datasets
* Hyperparameter optimization
* API development for ML models
* Deployment-ready architecture

---

## 📦 Run Locally

```bash
pip install -r requirements.txt
python src/train.py
uvicorn app:app --reload
```

---

## 🐳 Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

