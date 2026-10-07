import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------------------------------------
# 1. LOAD TRAINED MODEL
# ------------------------------------------------
model = xgb.XGBClassifier()
model.load_model("fraud_xgb_model.json")

# ------------------------------------------------
# 2. LOAD TEST FILE
# ------------------------------------------------
df = pd.read_csv("extra_4_200k.csv")

# ------------------------------------------------
# 3. SAME FEATURE ENGINEERING AS TRAIN
# ------------------------------------------------

df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

df["orig_diff_amount_ratio"] = df["orig_balance_diff"] / (df["amount"] + 1)
df["dest_diff_amount_ratio"] = df["dest_balance_diff"] / (df["amount"] + 1)

df["is_orig_zero_before"] = (df["oldbalanceOrg"] == 0).astype(int)
df["is_dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)

df["log_amount"] = np.log1p(df["amount"])

df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Drop unused
df = df.drop(columns=[
    "nameOrig",
    "nameDest",
    "isFlaggedFraud"
])

# ------------------------------------------------
# 4. SPLIT FEATURES / LABEL
# ------------------------------------------------
X_test = df.drop(columns=["isFraud"])
y_test = df["isFraud"]

# IMPORTANT: ensure same column order as training
X_test = X_test[model.get_booster().feature_names]

# ------------------------------------------------
# 5. PREDICTIONS
# ------------------------------------------------
threshold = 0.5

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# ------------------------------------------------
# 6. ML METRICS
# ------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# ------------------------------------------------
# 7. FINANCIAL METRICS
# ------------------------------------------------
df_eval = df.copy()
df_eval["pred"] = y_pred

total_fraud_amount = df_eval[df_eval["isFraud"] == 1]["amount"].sum()

flagged_df = df_eval[df_eval["pred"] == 1]

total_money_flagged = flagged_df["amount"].sum()

money_correctly_flagged = df_eval[
    (df_eval["pred"] == 1) & (df_eval["isFraud"] == 1)
]["amount"].sum()

money_false_flagged = df_eval[
    (df_eval["pred"] == 1) & (df_eval["isFraud"] == 0)
]["amount"].sum()

money_missed = df_eval[
    (df_eval["pred"] == 0) & (df_eval["isFraud"] == 1)
]["amount"].sum()

capture_rate = (
    money_correctly_flagged / total_fraud_amount
    if total_fraud_amount > 0 else 0
)

power_efficiency = (
    money_correctly_flagged / total_money_flagged
    if total_money_flagged > 0 else 0
)

# ------------------------------------------------
# 8. PRINT RESULTS
# ------------------------------------------------
print("------ ML METRICS ------")
print("Accuracy:", round(accuracy, 4))
print("AUC:", round(auc, 4))
print("PR-AUC:", round(pr_auc, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

print("\n------ FINANCIAL METRICS ------")
print("Total Fraud Amount:", round(total_fraud_amount, 2))
print("Total Money Flagged:", round(total_money_flagged, 2))
print("Money Correctly Flagged:", round(money_correctly_flagged, 2))
print("Money False Flagged:", round(money_false_flagged, 2))
print("Money Missed:", round(money_missed, 2))

print("\n------ ECONOMIC EFFICIENCY ------")
print("Fraud Capture Rate (Money Recall):", round(capture_rate, 4))
print("Power Efficiency:", round(power_efficiency, 4))