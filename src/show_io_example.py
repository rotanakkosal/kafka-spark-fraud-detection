import pandas as pd
import numpy as np
import joblib
import os
from tabulate import tabulate # <-- Import the new library

# --- Configuration ---
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "..", "data", "PS_20174392719_1491204439457_log.csv")
model_path = os.path.join(base_path, "..", "models", "fraud_model_robust.pkl")
scaler_path = os.path.join(base_path, "..", "models", "scaler_robust.pkl")

def print_table(df, title):
    """Utility function to print a DataFrame in a clean grid format."""
    print(title)
    # The 'psql' format gives a nice grid layout like the image
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

print("="*80)
print("🔍 SHOWCASING MODEL INPUT AND OUTPUT")
print("="*80)

# 1. LOAD A SMALL, CONSISTENT SAMPLE OF RAW DATA
df_sample = pd.read_csv(csv_path).sample(5, random_state=42)
print_table(df_sample[['step', 'type', 'amount', 'oldbalanceOrg', 'isFraud']], "\n[1] RAW DATA SAMPLE")

# 2. PERFORM THE DATA PROCESSING PIPELINE
# (This step is just text)
print("\n[2] Applying the full Data Processing Pipeline...")
print("   - Feature Engineering complete.")
print("   - Feature Scaling complete.")

# --- Perform the processing ---
features = pd.DataFrame()
features['step'] = df_sample['step']
features['amount'] = df_sample['amount']
features['log_amount'] = np.log1p(df_sample['amount'])
features['oldbalance_orig'] = df_sample['oldbalanceOrg']
features['oldbalance_dest'] = df_sample['oldbalanceDest']
features['balance_ratio'] = df_sample['amount'] / (df_sample['oldbalanceOrg'] + 1)
features['type_TRANSFER'] = (df_sample['type'] == 'TRANSFER').astype(int)
features['type_CASH_OUT'] = (df_sample['type'] == 'CASH_OUT').astype(int)

try:
    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)
except FileNotFoundError:
    print("\n❌ ERROR: Scaler file not found! Please run the trainer first.")
    exit()

# 3. SHOW THE FINAL MODEL INPUT
final_input_df = pd.DataFrame(features_scaled, columns=features.columns)
print_table(final_input_df.round(4), "\n[3] FINAL MODEL INPUT (Scaled Data)")
print("   Notice how all values are now on a similar scale (centered around 0).")

# 4. LOAD THE TRAINED MODEL AND PREDICT
print("\n[4] Loading the trained model and making predictions...")
try:
    model = joblib.load(model_path)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    print("   - Predictions complete.")
except FileNotFoundError:
    print("\n❌ ERROR: Model file not found! Please run the trainer first.")
    exit()

# 5. SHOW THE FINAL MODEL OUTPUT
output_df = df_sample[['amount', 'isFraud']].copy()
# Reset index to align with predictions if needed
output_df = output_df.reset_index(drop=True) 
output_df['PREDICTED_CLASS'] = predictions
output_df['FRAUD_PROBABILITY'] = probabilities[:, 1]
output_df['FRAUD_PROBABILITY'] = output_df['FRAUD_PROBABILITY'].map('{:.2%}'.format)

print_table(output_df, "\n[5] FINAL MODEL OUTPUT")
print("\n" + "="*80)