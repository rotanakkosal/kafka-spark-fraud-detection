import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 1. SETUP
base_path = os.path.dirname(__file__)
csv_path = '/home/data/kosal_moved_files/cbnu_assignment/realtime_fraud_detection/data/PS_20174392719_1491204439457_log.csv'
# --- Point to the new ROBUST model and scaler ---
model_path = os.path.join(base_path, "..", "..", "models", "fraud_model_robust.pkl")
scaler_path = os.path.join(base_path, "..", "..", "models", "scaler_robust.pkl")


print("="*60)
print("🔍 ROBUST TRANSACTION-LEVEL MODEL VALIDATION")
print("="*60)

# 2. LOAD TEST DATA (different sample from training)
print("\n>> Loading test data...")
try:
    df = pd.read_csv(csv_path).sample(frac=0.2, random_state=99)  # Different seed
    print(f"   Loaded {len(df):,} transactions")
except FileNotFoundError:
    print(f"❌ ERROR: {csv_path} not found!")
    exit(1)

fraud_count = df['isFraud'].sum()
print(f"   Fraud: {fraud_count:,} ({df['isFraud'].mean()*100:.2f}%)")
print(f"   Normal: {len(df)-fraud_count:,}")

# 3. FEATURE ENGINEERING (Must match robust training exactly!)
print("\n>> Engineering features (non-leaky)...")

# --- Remove the leaky features ---
features = pd.DataFrame()
features['step'] = df['step']
features['amount'] = df['amount']
features['log_amount'] = np.log1p(df['amount'])
features['oldbalance_orig'] = df['oldbalanceOrg']
features['oldbalance_dest'] = df['oldbalanceDest']
features['balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
features['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
features['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)


features = features.replace([np.inf, -np.inf], 0).fillna(0)
y_true = df['isFraud'].values

# 4. LOAD MODEL AND SCALER
print("\n>> Loading robust model and scaler...")
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("   ✅ Model and scaler loaded successfully")
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("   Please run src/model_trainer/trainer_robust.py first!")
    exit(1)

# 5. SCALE AND PREDICT
X_scaled = scaler.transform(features)
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of fraud

# 6. EVALUATION
print("\n" + "="*60)
print("📊 EVALUATION RESULTS")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📈 Confusion Matrix:")
print(f"                    Predicted")
print(f"                  Normal  Fraud")
print(f"Actual  Normal    {tn:7d}  {fp:7d}")
print(f"        Fraud     {fn:7d}  {tp:7d}")

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_true, y_pred_proba)

print(f"\n📊 Key Metrics:")
print(f"   Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision:       {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall (TPR):    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:        {f1:.4f}")
print(f"   AUC-ROC:         {auc_roc:.4f}")

print(f"\n🎯 Detection Summary:")
print(f"   True Positives:   {tp:7d} (Fraud caught)")
print(f"   False Positives:  {fp:7d} (False alarms)")
print(f"   False Negatives:  {fn:7d} (Missed fraud)")
print(f"   True Negatives:   {tn:7d} (Correct normal)")

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"\n   False Positive Rate: {fpr*100:.2f}%")
print(f"   False Negative Rate: {fnr*100:.2f}%")

print("\n" + "-"*60)
print("\n📋 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud'], digits=4))

# Insights
print("\n💡 Performance Assessment:")
if precision > 0.9 and recall > 0.9:
    print("   ✅ Excellent performance!")
elif precision > 0.7 and recall > 0.7:
    print("   ✅ Good performance - production ready")
elif precision > 0.5 or recall > 0.6:
    print("   ⚠️  Moderate performance - needs tuning")
else:
    print("   ❌ Poor performance - needs major improvements")

if precision < 0.5:
    print("   ⚠️  Low precision - many false alarms")
if recall < 0.6:
    print("   ⚠️  Low recall - missing too much fraud")

print("\n" + "="*60)
print("✅ VALIDATION COMPLETE")
print("="*60)
