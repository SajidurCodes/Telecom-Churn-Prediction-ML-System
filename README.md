# Telecom Churn Prediction ML System

An end-to-end machine learning project for predicting customer churn using LightGBM, stratified cross-validation, class imbalance handling, and feature engineering.

## Project Overview

This project predicts whether a telecom customer is likely to churn. It was built as part of a Kaggle-style ML  journey and structured as a production-style machine learning project.

## Features

- LightGBM binary classification model
- Stratified K-Fold cross-validation
- Class imbalance handling using `scale_pos_weight`
- Reusable feature engineering pipeline
- Saved model artifact for inference
- Separate training and prediction scripts

## Final Performance

- **Final CV ROC-AUC:** 0.8995

## Project Structure

```bash
telecom-churn-prediction-ml-system/
├── data/
├── models/
├── src/
├── requirements.txt
├── README.md
└── .gitignore