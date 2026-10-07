import pandas as pd
import numpy as np
import time
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

start_time = time.time()

# --- Load Data ---
df = pd.read_csv("extra_4_200k.csv")

# --- Define Rule-Based Fraud System ---
def fraud_rule_combined(row):

    if row["type"] not in ["TRANSFER", "CASH_OUT"]:
        return 0

    score = 0

    # Rule 1: Large transaction
    if row["amount"] > 200000:
        score += 1

    # Rule 2: Sender drained
    if row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0:
        score += 2

    # Rule 3: Accounting mismatch
    expected = row["oldbalanceOrg"] - row["amount"]
    if abs(expected - row["newbalanceOrig"]) > 1:
        score += 2

    # Rule 4: Destination was empty
    if row["oldbalanceDest"] == 0:
        score += 1

    # Threshold
    if score >= 3:
        return 1

    return 0


# --- Apply Rule ---
df["rule_pred"] = df.apply(fraud_rule_combined, axis=1)

y_true = df["isFraud"]
y_pred = df["rule_pred"]

# --- Performance Metrics ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

print("===== RULE-BASED MODEL METRICS =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

# --- Financial Impact Analysis ---
results_df = df.copy()

tp = (results_df["isFraud"] == 1) & (results_df["rule_pred"] == 1)
fp = (results_df["isFraud"] == 0) & (results_df["rule_pred"] == 1)
fn = (results_df["isFraud"] == 1) & (results_df["rule_pred"] == 0)

money_saved = results_df.loc[tp, "amount"].sum()
money_wrongly_flagged = results_df.loc[fp, "amount"].sum()
money_lost = results_df.loc[fn, "amount"].sum()
total_transaction_amount = results_df["amount"].sum()
money_flagged_total = results_df.loc[results_df["rule_pred"] == 1, "amount"].sum()

print("\n===== FINANCIAL IMPACT =====")
print(f"Total transaction amount processed: {total_transaction_amount:,.2f}")
print(f"Money saved (True Fraud Caught): {money_saved:,.2f}")
print(f"Money wrongly flagged (False Positives): {money_wrongly_flagged:,.2f}")
print(f"Money lost (Fraud Missed): {money_lost:,.2f}")
print(f"Total money flagged by rule: {money_flagged_total:,.2f}")

print("\n===== EVENT COUNTS =====")
print(f"Fraud caught (TP): {tp.sum()}")
print(f"False alarms (FP): {fp.sum()}")
print(f"Fraud missed (FN): {fn.sum()}")

# --- Efficiency Metrics ---
end_time = time.time()
duration_sec = end_time - start_time

cpu_power_watts = 65  # adjust if needed
energy_kwh = cpu_power_watts * duration_sec / 3600 / 1000

print("\n===== EFFICIENCY =====")
print(f"Time taken: {duration_sec:.4f} seconds")
print(f"Approx. Electricity used: {energy_kwh:.6f} kWh (~{energy_kwh*1000:.4f} Wh)")