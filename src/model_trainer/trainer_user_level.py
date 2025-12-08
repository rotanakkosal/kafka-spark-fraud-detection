import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# 1. SETUP
# Get the path to the CSV file (assumed to be in the same folder)
base_path = os.path.dirname(__file__)
csv_path = '/home/data/kosal_moved_files/cbnu_assignment/realtime_fraud_detection/data/PS_20174392719_1491204439457_log.csv'
model_path = os.path.join(base_path, "isolation_forest.pkl")

print(f"Reading data from: {csv_path} (This may take a minute...)")

# 2. LOAD & PREPROCESS REAL DATA
try:
    # Read only 10% of data to save memory/time for the assignment
    df = pd.read_csv(csv_path).sample(frac=0.1, random_state=42)
except FileNotFoundError:
    print(f"\n❌ ERROR: {os.path.basename(csv_path)} not found!")
    print(f"Please check the path: {csv_path}")
    exit(1)

print(">> Aggregating data to create features (Tries, Amount, Fail Rate)...")

# We need to turn raw transactions into 'User Behavior' features
# In PaySim: 
# 'nameOrig' is the User ID.
# 'amount' is the transaction amount.
# 'isFraud' is our label (useful to check, but we train unsupervised).

# Group by User to mimic your Spark Window logic
user_features = df.groupby('nameOrig').agg({
    'amount': ['count', 'sum'],         # tries, total_amount
    'isFraud': 'mean'                   # We use this as a proxy for 'fail_rate' or risk
}).reset_index()

# Flatten columns
user_features.columns = ['user_id', 'tries', 'total_amount', 'risk_factor']

# Create a 'geo_mismatch' feature (Simulated, as PaySim doesn't have country codes)
# We assume 5% of users travel
user_features['geo_mismatch'] = np.random.choice([0, 1], size=len(user_features), p=[0.95, 0.05])

# Select features for training
features = ['tries', 'total_amount', 'risk_factor', 'geo_mismatch']
X_train = user_features[features]

print(f">> Training on {len(X_train)} real user profiles...")

# 3. TRAIN MODEL
# We use contamination=0.01 because real fraud is rare
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_train)

# 4. SAVE MODEL
joblib.dump(model, model_path)
print(f"✅ Trained on Real Data! Model saved to: {model_path}")
print("You can now run 'spark-submit src/detector.py' to use this new brain.")