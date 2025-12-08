import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

base_path = os.path.dirname(os.path.dirname(__file__))  # Go up to realtime_fraud_detection directory
csv_path = '/home/data/kosal_moved_files/cbnu_assignment/realtime_fraud_detection/data/PS_20174392719_1491204439457_log.csv'
model_path = os.path.join(base_path, "models", "fraud_model_transaction.pkl")
scaler_path = os.path.join(base_path, "models", "scaler_transaction.pkl")

print("="*60)
print("🔍 OVERFITTING DIAGNOSIS")
print("="*60)

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def engineer_features(df):
    """Feature engineering function"""
    features = pd.DataFrame()
    features['amount'] = df['amount']
    features['step'] = df['step']
    features['oldbalance_orig'] = df['oldbalanceOrg']
    features['newbalance_orig'] = df['newbalanceOrig']
    features['oldbalance_dest'] = df['oldbalanceDest']
    features['newbalance_dest'] = df['newbalanceDest']
    features['balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    features['balance_to_zero'] = (df['newbalanceOrig'] == 0).astype(int)
    features['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
    features['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    features['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    features['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)
    features['type_CASH_IN'] = (df['type'] == 'CASH_IN').astype(int)
    features['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
    features['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    features['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
    features['log_amount'] = np.log1p(df['amount'])
    return features.replace([np.inf, -np.inf], 0).fillna(0)

# Test on 3 different samples
print("\n>> Testing on multiple data samples...")
results = []

for i, seed in enumerate([42, 99, 123], 1):
    print(f"\n   Sample {i} (seed={seed}):")
    df = pd.read_csv(csv_path).sample(frac=0.1, random_state=seed)
    
    features = engineer_features(df)
    X_scaled = scaler.transform(features)
    y_true = df['isFraud'].values
    
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    print(f"      AUC-ROC:   {auc:.4f}")
    
    results.append({
        'sample': i,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    })

# Check variance
df_results = pd.DataFrame(results)
print("\n" + "="*60)
print("📊 VARIANCE ANALYSIS")
print("="*60)
print(f"\nPrecision: {df_results['precision'].mean():.4f} ± {df_results['precision'].std():.4f}")
print(f"Recall:    {df_results['recall'].mean():.4f} ± {df_results['recall'].std():.4f}")
print(f"F1-Score:  {df_results['f1'].mean():.4f} ± {df_results['f1'].std():.4f}")
print(f"AUC-ROC:   {df_results['auc'].mean():.4f} ± {df_results['auc'].std():.4f}")

print("\n💡 Diagnosis:")
if df_results['precision'].std() > 0.05 or df_results['recall'].std() > 0.05:
    print("   ⚠️  HIGH VARIANCE - Model is likely overfitting!")
    print("   Performance varies significantly across samples")
else:
    print("   ✅ LOW VARIANCE - Model generalizes well")

if df_results['precision'].mean() > 0.95:
    print("   ⚠️  SUSPICIOUSLY HIGH PRECISION - Check for data leakage!")
