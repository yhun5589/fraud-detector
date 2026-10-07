import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# 1. Load Model & Raw Data
# =========================
print("[1/4] Loading model and test dataset...")
model = joblib.load("lightgbm_fraud_model.pkl")
df = pd.read_csv("extra_4_200k.csv")

# =========================
# 2. Replicate Feature Engineering
# =========================
print("[2/4] Applying matching feature engineering pipeline...")

# Filter relevant transaction vectors (Fraud only occurs on TRANSFER & CASH_OUT)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

# Feature Engineering
df["is_merchant_dest"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
df["error_balance_orig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
df["is_liquidation"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)

df["orig_tx_count"] = df.groupby("nameOrig")["amount"].transform("count")
df["dest_tx_count"] = df.groupby("nameDest")["amount"].transform("count")

# Drop raw identifiers & set categorical types
df = df.drop(columns=["nameOrig", "nameDest"])
df["type"] = df["type"].astype("category")

# Separate features, target, and dollar values
X = df.drop(columns=["isFraud"])
y = df["isFraud"]
amount = df["amount"]

# =========================
# 3. Inference & Thresholding
# =========================
print("[3/4] Running inference...")
y_prob = model.predict(X)

# Set decision threshold (Use optimal threshold from training, e.g., 0.35 or tune on test)
# For calibrated probability outputs without scale_pos_weight, start at 0.5 or lower.
OPTIMAL_THRESHOLD = 0.5 
y_pred = (y_prob >= OPTIMAL_THRESHOLD).astype(int)

# =========================
# 4. ML Metrics
# =========================
print("[4/4] Calculating metrics...\n")

accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
pr_auc = average_precision_score(y, y_prob)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

# =========================
# 5. Financial & Economic Metrics
# =========================
total_fraud_amount = amount[y == 1].sum()
total_money_flagged = amount[y_pred == 1].sum()
money_correctly_flagged = amount[(y == 1) & (y_pred == 1)].sum()
money_false_flagged = amount[(y == 0) & (y_pred == 1)].sum()
money_missed = amount[(y == 1) & (y_pred == 0)].sum()

capture_rate = (
    money_correctly_flagged / total_fraud_amount
    if total_fraud_amount > 0 else 0.0
)

power_efficiency = (
    money_correctly_flagged / total_money_flagged
    if total_money_flagged > 0 else 0.0
)

# =========================
# 6. Report
# =========================
print("=" * 45)
print("             EVALUATION REPORT")
print("=" * 45)
print("------ ML METRICS ------")
print(f"Accuracy:                  {accuracy:.4f}")
print(f"ROC-AUC:                   {auc:.4f}")
print(f"PR-AUC (Avg Precision):    {pr_auc:.4f}")
print(f"Precision:                 {precision:.4f}")
print(f"Recall:                    {recall:.4f}")
print(f"F1 Score:                  {f1:.4f}")
print(f"Confusion Matrix:          TP={tp} | FP={fp} | FN={fn} | TN={tn}")

print("\n------ FINANCIAL METRICS ------")
print(f"Total Fraud Exposure:     ${total_fraud_amount:,.2f}")
print(f"Total Money Flagged:       ${total_money_flagged:,.2f}")
print(f"Money Correctly Recovered: ${money_correctly_flagged:,.2f}")
print(f"Money False Flagged (Friction): ${money_false_flagged:,.2f}")
print(f"Money Missed (Direct Loss):      ${money_missed:,.2f}")

print("\n------ ECONOMIC EFFICIENCY ------")
print(f"Fraud Capture Rate (Dollar Recall): {capture_rate * 100:.2f}%")
print(f"Power Efficiency (Dollar Precision):{power_efficiency * 100:.2f}%")
print("=" * 45)