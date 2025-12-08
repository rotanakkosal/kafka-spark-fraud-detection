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
    f1_score
)

# 1. SETUP PATHS
base_path = os.path.dirname(__file__)
csv_path = '/home/data/kosal_moved_files/cbnu_assignment/realtime_fraud_detection/data/PS_20174392719_1491204439457_log.csv'
model_path = os.path.join(base_path, "isolation_forest.pkl")

# 2. LOAD DATA (We use a different random sample for testing)
print(">> Loading data for validation...")
try:
    # Read 20% of data for testing
    df = pd.read_csv(csv_path).sample(frac=0.2, random_state=99)
    if df.empty:
        print("❌ ERROR: Loaded dataset is empty!")
        exit(1)
    print(f"   Loaded {len(df):,} transactions")
except FileNotFoundError:
    print(f"❌ ERROR: {csv_path} not found!")
    exit(1)
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    exit(1)

# 3. PREPROCESS (Must match training exactly!)
print(">> Aggregating user behavior...")

# Validate required columns exist
required_cols = ['nameOrig', 'amount', 'isFraud']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ ERROR: Missing required columns: {missing_cols}")
    exit(1)

# Group by User to create the same features we trained on
user_features = df.groupby('nameOrig').agg({
    'amount': ['count', 'sum'],
    'isFraud': ['mean', 'max']  # Get risk factor and true label
}).reset_index()

# Flatten columns
user_features.columns = ['user_id', 'tries', 'total_amount', 'risk_factor', 'is_fraud']

# Add the simulated geo_mismatch (Randomized same as training)
# IMPORTANT: Set seed for reproducibility
np.random.seed(42)  # Match training seed
user_features['geo_mismatch'] = np.random.choice(
    [0, 1], 
    size=len(user_features), 
    p=[0.95, 0.05]
)

# Prepare Features (X) and Labels (y)
features = ['tries', 'total_amount', 'risk_factor', 'geo_mismatch']  # Must match training!
X_test = user_features[features]
y_true = user_features['is_fraud'].astype(int)  # Ensure binary

# Validate features
if X_test.isnull().any().any():
    print("⚠️  WARNING: Features contain NaN values. Filling with 0...")
    X_test = X_test.fillna(0)

print(f"   Test users: {len(X_test):,}")
print(f"   Fraud users: {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
print(f"   Normal users: {(y_true==0).sum():,}")

# 4. PREDICT USING YOUR MODEL
print(">> Loading model and predicting...")
try:
    model = joblib.load(model_path)
    print("   ✅ Model loaded successfully")
except FileNotFoundError:
    print(f"❌ ERROR: Model not found at {model_path}")
    print("   Please run models/trainer.py first to train the model!")
    exit(1)
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit(1)

# Isolation Forest returns: 1 (Normal), -1 (Anomaly/Fraud)
# Convert to: 0 (Normal), 1 (Fraud) to match PaySim labels
y_pred_raw = model.predict(X_test)
y_pred = np.array([1 if x == -1 else 0 for x in y_pred_raw])

# 5. GENERATE REPORT
print("\n" + "="*60)
print("       MODEL VALIDATION REPORT")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📈 Confusion Matrix:")
print(f"                    Predicted")
print(f"                  Normal  Fraud")
print(f"Actual  Normal      {tn:5d}  {fp:5d}")
print(f"        Fraud       {fn:5d}  {tp:5d}")

# Key Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n📊 Key Metrics:")
print(f"   Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision:       {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall (TPR):    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:        {f1:.4f}")
print(f"   Specificity:      {specificity:.4f} ({specificity*100:.2f}%)")

print(f"\n🎯 Fraud Detection Summary:")
print(f"   Total Users Tested:     {len(y_true):,}")
print(f"   True Positives (TP):    {tp:5d} (Fraud detected correctly)")
print(f"   False Positives (FP):   {fp:5d} (False alarms)")
print(f"   False Negatives (FN):   {fn:5d} (Missed fraud)")
print(f"   True Negatives (TN):    {tn:5d} (Correctly normal)")

print(f"\n   False Positive Rate:   {fp/(fp+tn)*100:.2f}%")
print(f"   False Negative Rate:   {fn/(fn+tp)*100:.2f}%")

print("-" * 60)

# Classification Report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud'], digits=4))

# Additional insights
print("\n💡 Insights:")
if precision < 0.1:
    print("   ⚠️  Very low precision - too many false alarms!")
if recall < 0.5:
    print("   ⚠️  Low recall - missing many fraud cases!")
if f1 < 0.3:
    print("   ⚠️  Overall performance needs improvement")
if precision > 0.7 and recall > 0.7:
    print("   ✅ Good balance between precision and recall")