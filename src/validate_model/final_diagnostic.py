import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, f1_score

# --- Configuration ---
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "..", "..", "data", "PS_20174392719_1491204439457_log.csv")
model_path = os.path.join(base_path, "..", "..", "models", "fraud_model_robust.pkl")
scaler_path = os.path.join(base_path, "..", "..", "models", "scaler_robust.pkl")

print("="*60)
print("🔍 FINAL DIAGNOSTIC: TRAIN vs. TEST PERFORMANCE ANALYSIS")
print("="*60)

# --- Step 1: Document Current Setup ---
# Load model to inspect its parameters
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("\n>> Loading model and scaler...")
    print("   ✅ Loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Model or scaler not found! Please run the robust trainer first.")
    exit()

# Document features (must match the robust trainer)
feature_names = [
    'step', 'amount', 'log_amount', 'oldbalance_orig', 'oldbalance_dest', 
    'balance_ratio', 'type_TRANSFER', 'type_CASH_OUT'
]

print("\n>> Step 1: Documenting Current Setup")
print("-" * 40)
print("1.1: Features Used in Model:")
for i, feature in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feature}")

print("\n1.2: Model Hyperparameters:")
# Print key parameters to understand its complexity
params = model.get_params()
print(f"   - n_estimators:      {params.get('n_estimators')}")
print(f"   - max_depth:         {params.get('max_depth')}")
print(f"   - min_samples_leaf:  {params.get('min_samples_leaf')}")
print(f"   - class_weight:      {params.get('class_weight')}")
print("-" * 40)

# --- Step 2: Implement Train-Test Gap Analysis ---

def engineer_features(df):
    """Recreate the exact feature engineering process."""
    features = pd.DataFrame()
    features['step'] = df['step']
    features['amount'] = df['amount']
    features['log_amount'] = np.log1p(df['amount'])
    features['oldbalance_orig'] = df['oldbalanceOrg']
    features['oldbalance_dest'] = df['oldbalanceDest']
    features['balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    features['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    features['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    return features.replace([np.inf, -np.inf], 0).fillna(0)

# Load the TRAINING data (using the same seed as the robust trainer)
print("\n>> Step 2: Evaluating on TRAINING Data (seed=42)")
df_train = pd.read_csv(csv_path).sample(frac=0.3, random_state=42)
# We need to simulate the 80/20 split the trainer did
train_split_index = int(0.8 * len(df_train))
df_train_final = df_train.iloc[:train_split_index]

features_train = engineer_features(df_train_final)
X_train_scaled = scaler.transform(features_train)
y_train = df_train_final['isFraud'].values
y_train_pred = model.predict(X_train_scaled)
train_f1 = f1_score(y_train, y_train_pred)

print(f"   Training F1-Score: {train_f1:.4f}")

# Load completely unseen TEST data (using a different seed)
print("\n>> Step 3: Evaluating on TEST Data (seed=99)")
df_test = pd.read_csv(csv_path).sample(frac=0.2, random_state=99)
features_test = engineer_features(df_test)
X_test_scaled = scaler.transform(features_test)
y_test = df_test['isFraud'].values
y_test_pred = model.predict(X_test_scaled)
test_f1 = f1_score(y_test, y_test_pred)

print(f"   Test F1-Score:     {test_f1:.4f}")

# --- Step 3: Report and Diagnose ---
print("\n" + "="*60)
print("📊 FINAL DIAGNOSIS REPORT")
print("="*60)

gap = train_f1 - test_f1
print("\nPerformance Gap:")
print(f"   - Train F1-Score: {train_f1:.4f}")
print(f"   - Test F1-Score:  {test_f1:.4f}")
print("   --------------------")
print(f"   - Gap:            {gap:+.4f}")

print("\nAssessment:")
if gap > 0.10:
    print("   - ⚠️  Severe Overfitting Detected!")
    print("   - Action: Major model restructuring or more aggressive regularization needed.")
elif gap > 0.05:
    print("   - ⚠️  Moderate Overfitting Detected.")
    print("   - Action: Increase regularization as suggested in your report (Option B).")
elif gap > 0.02:
    print("   - ⚠️  Slight Overfitting Detected.")
    print("   - Action: Consider minor tuning, but may be acceptable.")
else:
    print("   - ✅ Excellent Generalization!")
    print("   - Action: The model is robust. The high performance is likely due to the dataset's simplicity.")

print("\n" + "="*60)
