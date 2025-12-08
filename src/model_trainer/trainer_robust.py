import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# --- Configuration ---
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "..", "..", "data", "PS_20174392719_1491204439457_log.csv")
model_path = os.path.join(base_path, "..", "..", "models", "fraud_model_robust.pkl")
scaler_path = os.path.join(base_path, "..", "..", "models", "scaler_robust.pkl")

print("="*60)
print("🔍 ROBUST TRANSACTION-LEVEL MODEL (Anti-Leakage & Regularized)")
print("="*60)

# 1. LOAD DATA
print(f"\n>> Loading data...")
# Use a larger sample for more robust training
df = pd.read_csv(csv_path).sample(frac=0.3, random_state=42) 
print(f"   Loaded {len(df):,} transactions")

# 2. FEATURE ENGINEERING (WITHOUT LEAKY FEATURES)
print("\n>> Engineering features using only pre-transaction data...")

# These are the features you would know BEFORE approving a transaction
features = pd.DataFrame()
features['step'] = df['step']
features['amount'] = df['amount']
features['log_amount'] = np.log1p(df['amount'])
features['oldbalance_orig'] = df['oldbalanceOrg']
features['oldbalance_dest'] = df['oldbalanceDest']
features['balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1) # Still a powerful, valid feature
features['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
features['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)

# Target variable
y = df['isFraud'].values

print(f"   Using {len(features.columns)} non-leaky features.")

# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
print("\n>> Splitting data into 80% train / 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {len(X_train):,} transactions ({y_train.sum()} fraud)")
print(f"   Test set:     {len(X_test):,} transactions ({y_test.sum()} fraud)")

# 4. SCALE FEATURES
print("\n>> Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. TRAIN MODEL WITH REGULARIZATION
print("\n>> Training a regularized Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,             # Limits tree depth to prevent overfitting
    min_samples_leaf=10,      # Ensures leaves aren't too specific
    class_weight='balanced',  # Handles the rare fraud cases
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# 6. EVALUATE ON THE UNSEEN TEST SET
print("\n" + "="*60)
print("📊 FINAL PERFORMANCE ON UNSEEN TEST DATA")
print("="*60)
y_pred = model.predict(X_test_scaled)

print("\n📋 Classification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4))

# 7. FEATURE IMPORTANCE
print("\n>> Top Features (No Leakage):")
feature_importance = pd.DataFrame({
    'feature': features.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# 8. SAVE THE ROBUST MODEL AND SCALER
print("\n>> Saving the new, robust model and scaler...")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"   ✅ Model saved to: {model_path}")
print(f"   ✅ Scaler saved to: {scaler_path}")

print("\n" + "="*60)
print("✅ ROBUST TRAINING COMPLETE!")
print("="*60)
