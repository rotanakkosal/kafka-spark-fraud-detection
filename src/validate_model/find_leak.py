import pandas as pd
import joblib
import os

# Define paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(base_path, "models", "fraud_model_transaction.pkl")

print("="*60)
print("🔍 FINDING DATA LEAKAGE: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Define the feature names in the exact order they were trained
# This is crucial for matching importance scores to names
feature_names = [
    'amount', 'step', 'oldbalance_orig', 'newbalance_orig', 'oldbalance_dest', 
    'newbalance_dest', 'balance_ratio', 'balance_to_zero', 'type_PAYMENT', 
    'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN', 
    'dest_is_merchant', 'balance_change_orig', 'balance_change_dest', 'log_amount'
]

# Load the trained model
try:
    model = joblib.load(model_path)
    print(f"\n>> Successfully loaded model from: {model_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Model not found at {model_path}")
    print("   Please ensure the model exists.")
    exit()
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit()

# Get feature importances from the model
importances = model.feature_importances_

# Create a DataFrame to view the results
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n>> Top 10 Most Important Features:")
print("-" * 40)
print(feature_importance_df.head(10).to_string(index=False))
print("-" * 40)

# Analyze the top feature
top_feature = feature_importance_df.iloc[0]
print("\n💡 Diagnosis:")
print(f"   The most important feature is '{top_feature['feature']}' with an importance of {top_feature['importance']:.4f}.")

if top_feature['importance'] > 0.5:
    print(f"\n   ⚠️  DATA LEAK SUSPECTED!")
    print(f"   The feature '{top_feature['feature']}' is extremely predictive. It might contain information that is too closely related to the fraud label itself.")
    print(f"   This often happens with features calculated from balances or transaction outcomes that wouldn't be known *before* a transaction is approved in a real-world scenario.")
else:
    print("\n   ✅ No single feature dominates completely. The model is likely using a combination of signals, which is healthier.")

print("\n" + "="*60)
