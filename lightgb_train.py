import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ==========================================
# 1. Load Data
# ==========================================
print("[1/6] Loading data...")
df = pd.read_csv("train_300k.csv")

# ==========================================
# 2. Domain-Specific Filtering
# ==========================================
print("[2/6] Filtering irrelevant transaction types...")
# Fraud in financial accounting datasets (e.g. PaySim) only occurs on TRANSFER & CASH_OUT.
# Training on PAYMENT, CASH_IN, or DEBIT dilutes tree split quality.
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

# ==========================================
# 3. Feature Engineering
# ==========================================
print("[3/6] Engineering accounting anomaly features...")

# Flag destination entities (Merchant vs Customer)
df["is_merchant_dest"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

# Extract accounting balance mismatch errors
# Trees cannot compute multi-column mathematical differences natively
df["error_balance_orig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

# Account liquidation flag (Full account drain)
df["is_liquidation"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)

# Account interaction frequencies (Velocity features)
df["orig_tx_count"] = df.groupby("nameOrig")["amount"].transform("count")
df["dest_tx_count"] = df.groupby("nameDest")["amount"].transform("count")

# Drop raw non-numeric identifier strings
df = df.drop(columns=["nameOrig", "nameDest"])

# Encode categoricals for LightGBM
df["type"] = df["type"].astype("category")

# ==========================================
# 4. Train / Validation Split
# ==========================================
print("[4/6] Splitting dataset (Stratified)...")
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# ==========================================
# 5. Hyperparameters & Model Training
# ==========================================
print("[5/6] Training LightGBM model...")

# NOTE: scale_pos_weight is REMOVED to preserve true probability calibration.
# Objective optimization uses PR-AUC ('average_precision') directly.
params = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,              # Constrained to prevent over-fitting minority class
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbosity": -1,
    "seed": 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

# ==========================================
# 6. Prediction & Threshold Optimization
# ==========================================
print("[6/6] Evaluating & tuning decision threshold...")

y_pred_prob = model.predict(X_val)

# Optimize threshold based on Precision-Recall Curve to maximize F1-score
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
optimal_f1 = f1_scores[best_idx]

y_pred_optimal = (y_pred_prob >= best_threshold).astype(int)

# ==========================================
# Metrics Output
# ==========================================
print("\n" + "="*50)
print("             MODEL EVALUATION REPORT")
print("="*50)
print(f"ROC-AUC Score:             {roc_auc_score(y_val, y_pred_prob):.5f}")
print(f"PR-AUC (Avg Precision):    {average_precision_score(y_val, y_pred_prob):.5f}")
print(f"Optimal Decision Threshold:{best_threshold:.4f}")
print(f"Maximized F1 Score:        {optimal_f1:.5f}")
print("-" * 50)
print("\nClassification Report (At Optimal Threshold):")
print(classification_report(y_val, y_pred_optimal, digits=4))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred_optimal)
print(f"TN: {cm[0][0]:<7} | FP: {cm[0][1]}")
print(f"FN: {cm[1][0]:<7} | TP: {cm[1][1]}")

# ==========================================
# Feature Importance Ranking
# ==========================================
print("\nTop 10 Feature Importances:")
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importance(importance_type="gain")
}).sort_values(by="Importance", ascending=False)

print(importance.head(10).to_string(index=False))

# ==========================================
# Save Artifacts
# ==========================================
model.save_model("lightgbm_fraud_model.txt")
joblib.dump(model, "lightgbm_fraud_model.pkl")
print("\nArtifacts saved: 'lightgbm_fraud_model.txt', 'lightgbm_fraud_model.pkl'")