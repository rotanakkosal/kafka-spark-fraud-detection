import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "..", "..", "data", "PS_20174392719_1491204439457_log.csv")
model_path = os.path.join(base_path, "..", "..", "models", "fraud_model_robust.pkl")
scaler_path = os.path.join(base_path, "..", "..", "models", "scaler_robust.pkl")
output_dir = os.path.join(base_path, "..", "..", "report")
os.makedirs(output_dir, exist_ok=True) # Ensure the report directory exists

print("="*60)
print("🔍 VISUAL PERFORMANCE ANALYSIS OF ROBUST MODEL")
print("="*60)

# 1. LOAD DATA
print(f"\n>> Loading test data sample...")
df = pd.read_csv(csv_path).sample(frac=0.2, random_state=99)
print(f"   Loaded {len(df):,} transactions")

# 2. FEATURE ENGINEERING
print("\n>> Engineering non-leaky features...")
features = pd.DataFrame()
features['step'] = df['step']
features['amount'] = df['amount']
features['log_amount'] = np.log1p(df['amount'])
features['oldbalance_orig'] = df['oldbalanceOrg']
features['oldbalance_dest'] = df['oldbalanceDest']
features['balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
features['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
features['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
y_true = df['isFraud'].values

# 3. LOAD MODEL AND SCALER
print("\n>> Loading the robust model and scaler...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("   ✅ Model and scaler loaded successfully.")

# 4. SCALE FEATURES AND PREDICT
print("\n>> Scaling features and making predictions...")
X_scaled = scaler.transform(features)
y_pred = model.predict(X_scaled)
# Get probabilities for ROC and PR curves
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# 5. PRINT TEXT-BASED REPORT
print("\n" + "="*60)
print("📊 ROBUST MODEL PERFORMANCE (TEXT REPORT)")
print("="*60)
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud'], digits=4))

# 6. GENERATE VISUALIZATIONS
print("\n>> Generating performance visualizations...")

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Fraud Detection Model Performance Analysis', fontsize=20)

# --- Plot 1: Confusion Matrix Heatmap ---
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
axes[0, 0].set_title('Confusion Matrix', fontsize=16)
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

# --- Plot 2: ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
auc = roc_auc_score(y_true, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate (Recall)')
axes[0, 1].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
axes[0, 1].legend(loc="lower right")

# --- Plot 3: Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
axes[1, 0].plot(recall, precision, color='blue', lw=2)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve', fontsize=16)
axes[1, 0].grid(True)

# --- Plot 4: Feature Importance ---
importances = model.feature_importances_
feature_names = features.columns
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=axes[1, 1])
axes[1, 1].set_title('Feature Importance', fontsize=16)
axes[1, 1].set_xlabel('Importance Score')
axes[1, 1].set_ylabel('Features')

# Save the entire figure to a file
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = os.path.join(output_dir, "model_performance_visuals.png")
plt.savefig(output_path)

print(f"\n   ✅ Visualizations saved to: {output_path}")
print("\n" + "="*60)
print("✅ VISUALIZATION COMPLETE")
print("="*60)