import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load test data ---
test_df = pd.read_csv('test_200k.csv')

# --- 1. Basic Fraud Statistics ---
fraud_count = test_df['isFraud'].value_counts()
print("Fraud distribution:\n", fraud_count)

# Pie chart of fraud vs non-fraud
plt.figure(figsize=(6,6))
fraud_count.plot.pie(autopct='%1.1f%%', colors=['skyblue','salmon'], startangle=90, labels=['Non-Fraud','Fraud'])
plt.title('Fraud vs Non-Fraud Transactions')
plt.ylabel('')
plt.show()

# --- 2. Fraud by Transaction Type ---
type_fraud = test_df.groupby('type')['isFraud'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=type_fraud.index, y=type_fraud.values, palette='Reds_r')
plt.title('Number of Fraud Cases by Transaction Type')
plt.ylabel('Number of Frauds')
plt.xlabel('Transaction Type')
plt.show()

# --- 3. Amount Distribution: Fraud vs Non-Fraud ---
plt.figure(figsize=(8,5))
sns.kdeplot(test_df[test_df['isFraud']==0]['amount'], label='Non-Fraud', shade=True)
sns.kdeplot(test_df[test_df['isFraud']==1]['amount'], label='Fraud', shade=True)
plt.title('Transaction Amount Distribution: Fraud vs Non-Fraud')
plt.xlabel('Amount')
plt.ylabel('Density')
plt.xlim(0, 500_000)  # Focus on main range for visualization
plt.legend()
plt.show()

# --- 4. Correlation Heatmap ---
plt.figure(figsize=(10,8))
numeric_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']
corr = test_df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# --- 5. Top 10 Origin Accounts by Number of Frauds ---
top_origins = test_df[test_df['isFraud']==1]['nameOrig'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_origins.index, y=top_origins.values, palette='viridis')
plt.title('Top 10 Origin Accounts by Number of Frauds')
plt.ylabel('Number of Fraud Transactions')
plt.xlabel('Origin Account')
plt.xticks(rotation=45)
plt.show()

# --- 6. Step (Time) vs Fraud ---
plt.figure(figsize=(10,5))
sns.boxplot(x='isFraud', y='step', data=test_df)
plt.title('Transaction Step (Time) Distribution by Fraud')
plt.xlabel('Is Fraud')
plt.ylabel('Step')
plt.show()

# --- Optional: Fraud Ratio per Destination Account ---
dest_fraud_ratio = test_df.groupby('nameDest')['isFraud'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=dest_fraud_ratio.index, y=dest_fraud_ratio.values, palette='magma')
plt.title('Top 10 Destination Accounts by Fraud Ratio')
plt.ylabel('Fraud Ratio')
plt.xlabel('Destination Account')
plt.xticks(rotation=45)
plt.show()
