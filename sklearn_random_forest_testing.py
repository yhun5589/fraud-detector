import pandas as pd
import numpy as np
import joblib
import time
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score
)

# ============================================
# START TIMER
# ============================================
start_time = time.time()

# ============================================
# LOAD DATA
# ============================================
test_df = pd.read_csv("extra_4_200k.csv")

X_test = test_df.drop(columns=["isFraud", "isFlaggedFraud"])
y_test = test_df["isFraud"]

# ============================================
# LOAD TRAINED OBJECTS
# ============================================
clf = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ============================================
# ENCODE CATEGORICAL FEATURES
# ============================================
for col, le in label_encoders.items():
    class_mapping = {cls: i for i, cls in enumerate(le.classes_)}
    X_test[col] = X_test[col].map(class_mapping).fillna(-1).astype(int)

# ============================================
# SCALE NUMERIC FEATURES
# ============================================
numeric_cols = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ============================================
# PREDICTIONS
# ============================================
y_proba = clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# ============================================
# BASIC METRICS
# ============================================
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
fpr = fp / (fp + tn)

# ============================================
# FINANCIAL ANALYSIS
# ============================================
results_df = test_df.copy()
results_df["y_true"] = y_test.values
results_df["y_pred"] = y_pred
results_df["risk_score"] = y_proba

tp_mask = (results_df["y_true"] == 1) & (results_df["y_pred"] == 1)
fp_mask = (results_df["y_true"] == 0) & (results_df["y_pred"] == 1)
fn_mask = (results_df["y_true"] == 1) & (results_df["y_pred"] == 0)

money_saved = results_df.loc[tp_mask, "amount"].sum()
money_wrongly_flagged = results_df.loc[fp_mask, "amount"].sum()
money_lost = results_df.loc[fn_mask, "amount"].sum()

total_fraud_amount = results_df.loc[results_df["y_true"] == 1, "amount"].sum()
total_flagged_amount = results_df.loc[results_df["y_pred"] == 1, "amount"].sum()
total_transaction_amount = results_df["amount"].sum()

money_recall = money_saved / total_fraud_amount if total_fraud_amount > 0 else 0
money_precision = money_saved / total_flagged_amount if total_flagged_amount > 0 else 0
power_efficiency = money_precision

# ============================================
# TOP 1% CAPTURE
# ============================================
top_1pct_cutoff = int(len(results_df) * 0.01)
top_1pct = results_df.sort_values("risk_score", ascending=False).head(top_1pct_cutoff)

top1_money_captured = top_1pct.loc[top_1pct["y_true"] == 1, "amount"].sum()
top1_capture_rate = (
    top1_money_captured / total_fraud_amount
    if total_fraud_amount > 0 else 0
)

# ============================================
# COST-BASED MODEL
# ============================================
review_cost_per_tx = 5
fraud_loss_multiplier = 1.0

expected_loss = (fp * review_cost_per_tx) + (money_lost * fraud_loss_multiplier)

# ============================================
# ENERGY ESTIMATE
# ============================================
end_time = time.time()
duration_sec = end_time - start_time

cpu_power_watts = 65
energy_kwh = cpu_power_watts * duration_sec / 3600

# ============================================
# PRINT RESULTS
# ============================================

print("\n================ ML METRICS ================")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {auc:.4f}")
print(f"PR AUC: {pr_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\n================ FINANCIAL IMPACT ================")
print(f"Total Transaction Amount: {total_transaction_amount:,.2f}")
print(f"Total Fraud Amount: {total_fraud_amount:,.2f}")
print(f"Money Saved (TP): {money_saved:,.2f}")
print(f"Money Wrongly Flagged (FP): {money_wrongly_flagged:,.2f}")
print(f"Money Lost (FN): {money_lost:,.2f}")

print("\n================ ECONOMIC EFFICIENCY ================")
print(f"Money Recall (Fraud Capture Rate): {money_recall:.4f}")
print(f"Money Precision: {money_precision:.4f}")
print(f"Power Efficiency: {power_efficiency:.4f}")
print(f"Top 1% Money Capture Rate: {top1_capture_rate:.4f}")
print(f"Expected Operational + Fraud Loss: {expected_loss:,.2f}")

print("\n================ COMPUTE STATS ================")
print(f"Time Taken: {duration_sec:.2f} seconds")
print(f"Estimated Energy Used: {energy_kwh:.6f} kWh (~{energy_kwh*1000:.2f} Wh)")
