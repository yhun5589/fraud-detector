import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving the model

# --- Load training data ---
train_df = pd.read_csv('train_300k.csv')

# Separate features and target
X_train = train_df.drop(columns=['isFraud', 'isFlaggedFraud'])
y_train = train_df['isFraud']

# Encode categorical columns
label_encoders = {}
for col in ['nameOrig', 'nameDest', 'type']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    label_encoders[col] = le  # Save encoder for future predictions

# Scale numeric features
numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
clf.fit(X_train, y_train)

# Save the model and preprocessors
joblib.dump(clf, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model trained and saved!")

# --- Example: Predict on new data ---
def predict_transaction(new_data):
    """
    new_data: pd.DataFrame with same columns as X_train
    """
    # Encode categorical
    for col, le in label_encoders.items():
        new_data[col] = le.transform(new_data[col])
    
    # Scale numeric
    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    
    # Predict
    predictions = clf.predict(new_data)
    probabilities = clf.predict_proba(new_data)[:, 1]  # Probability of being fraud
    
    return predictions, probabilities

# Example usage:
# new_tx = pd.DataFrame([{
#     'step': 100, 'type': 'CASH_IN', 'amount': 5000, 'nameOrig': 'C1816283037',
#     'oldbalanceOrg': 10000, 'newbalanceOrig': 15000, 'nameDest': 'C1690031432',
#     'oldbalanceDest': 3000, 'newbalanceDest': 8000
# }])
# preds, probs = predict_transaction(new_tx)
# print(preds, probs)
