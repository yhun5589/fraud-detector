# Fraud Detection: XGBoost, LightGBM, and Random Forest Analysis

This repository contains the source code, training pipelines, and evaluation scripts for the research paper:

> **Comparative Performance Analysis of XGBoost, Random Forest and LightGBM for Fraud Detection**  
> *Author:* Korarich Kiattanaporn[cite: 1]

---

## 📌 Project Overview

This research evaluates three gradient-boosted and tree-based machine learning models (LightGBM, XGBoost, and Random Forest) alongside a legacy rule-based detection engine across 1,300,000 transaction records[cite: 1]. Models are benchmarked on classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) as well as financial impact (Capital Saved, False Flag Costs) and computational energy efficiency[cite: 1].

---

## 📁 Repository Structure

* **`lightgb_train.py`**: Model training pipeline using LightGBM.
* **`lightgb_eva.py`**: Evaluation script testing the trained LightGBM model against test datasets.
* **`xgboost_train.py`**: Model training pipeline using XGBoost.
* **`Xgboost_eva.py`**: Evaluation script testing the trained XGBoost model against test datasets.
* **`sklearn_random_forest_trainer.py`**: Model training pipeline using Scikit-Learn's Random Forest classifier.
* **`sklearn_random_forest_testing.py`**: Evaluation script testing the trained Random Forest model against test datasets.
* **`manual_test.py`**: Evaluation script testing by using deterministic rule-based systems.

---

## ⚙️ Requirements & Installation

Install the required Python packages before running the scripts:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm
