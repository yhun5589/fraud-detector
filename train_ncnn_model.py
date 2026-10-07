import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("train_300k.csv")

# Drop leakage columns
df = df.drop(columns=["isFlaggedFraud", "nameOrig", "nameDest"])

# Encode categorical
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# ============================================
# DEFINE LIGHTWEIGHT MLP
# ============================================
class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = FraudMLP(X_train.shape[1])

# ============================================
# TRAINING SETUP
# ============================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================
# TRAIN LOOP
# ============================================
epochs = 20

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_auc = roc_auc_score(y_val.numpy(), val_outputs.numpy())

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

# ============================================
# SAVE MODEL
# ============================================
torch.save(model.state_dict(), "fraud_mlp.pth")
print("Model saved as fraud_mlp.pth")

# Save scaler
import joblib
joblib.dump(scaler, "scaler_ncnn.pkl")
joblib.dump(label_encoders, "label_encoders_ncnn.pkl")