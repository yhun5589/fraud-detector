import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
df = pd.read_csv("train_300k.csv")

# ------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------

# Balance difference features
df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

# Ratio features (stability)
df["orig_diff_amount_ratio"] = df["orig_balance_diff"] / (df["amount"] + 1)
df["dest_diff_amount_ratio"] = df["dest_balance_diff"] / (df["amount"] + 1)

# Zero-balance flags
df["is_orig_zero_before"] = (df["oldbalanceOrg"] == 0).astype(int)
df["is_dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)

# Log amount (very important for tree stability)
df["log_amount"] = np.log1p(df["amount"])

# One-hot encode transaction type
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# ------------------------------------------------
# 3. DROP NON-USEFUL COLUMNS
# ------------------------------------------------
df = df.drop(columns=[
    "nameOrig",
    "nameDest",
    "isFlaggedFraud"
])

# ------------------------------------------------
# 4. TIME-BASED SPLIT (BETTER THAN RANDOM)
# ------------------------------------------------
df = df.sort_values("step")

split_step = df["step"].quantile(0.80)

train_df = df[df["step"] <= split_step]
test_df  = df[df["step"] > split_step]

X_train = train_df.drop(columns=["isFraud"])
y_train = train_df["isFraud"]

X_test = test_df.drop(columns=["isFraud"])
y_test = test_df["isFraud"]

# ------------------------------------------------
# 5. HANDLE CLASS IMBALANCE
# ------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ------------------------------------------------
# 6. TRAIN XGBOOST MODEL
# ------------------------------------------------
model = xgb.XGBClassifier(
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------------
# 7. EVALUATE
# ------------------------------------------------
y_pred = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_pred)

print("AUC:", round(auc, 4))
print("PR-AUC:", round(pr_auc, 4))

# ------------------------------------------------
# 8. SAVE MODEL
# ------------------------------------------------
model.save_model("fraud_xgb_model.json")
print("Model saved as fraud_xgb_model.json")