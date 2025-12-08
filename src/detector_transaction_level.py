from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import joblib

# 1. Setup Spark
spark = SparkSession.builder \
    .appName("FraudDetector_TransactionLevel") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 2. Load model and scaler
print(">> Loading model and scaler...")
try:
    model = joblib.load("models/fraud_model_robust.pkl")
    scaler = joblib.load("models/scaler_robust.pkl")
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# 3. Define ML Scoring Logic (Transaction-level UDF)
@udf(returnType=StringType())
def predict_fraud(amount, location, status, timestamp_str):
    """
    Predict fraud for a single transaction
    """
    try:
        # Engineer features (match training)
        features_dict = {
            'amount': float(amount),
            'step': 1,  # In real-time, you might calculate this from timestamp
            'oldbalance_orig': 0,  # These would come from your event if available
            'newbalance_orig': 0,
            'oldbalance_dest': 0,
            'newbalance_dest': 0,
            'balance_ratio': 0,  # Calculate if balance available
            'balance_to_zero': 0,
            'type_PAYMENT': 1 if status == 'success' else 0,
            'type_TRANSFER': 0,
            'type_CASH_OUT': 1 if status == 'fail' else 0,
            'type_DEBIT': 0,
            'type_CASH_IN': 0,
            'dest_is_merchant': 1 if location == 'KR' else 0,
            'balance_change_orig': 0,
            'balance_change_dest': 0,
            'log_amount': np.log1p(float(amount))
        }
        
        # Create DataFrame with correct feature order
        features_df = pd.DataFrame([features_dict])
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Predict
        pred = model.predict(features_scaled)[0]
        pred_proba = model.predict_proba(features_scaled)[0][1]
        
        # Return risk level with confidence
        if pred == 1:
            return f"HIGH_RISK ({pred_proba:.2%})"
        else:
            return f"Normal ({1-pred_proba:.2%})"
            
    except Exception as e:
        return f"Error: {str(e)}"

# 4. Read Kafka Stream
schema = StructType() \
    .add("user_id", StringType()) \
    .add("timestamp", StringType()) \
    .add("amount", DoubleType()) \
    .add("location", StringType()) \
    .add("status", StringType())

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "payment_attempts") \
    .load() \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))

# 5. Apply Transaction-Level Prediction (NO AGGREGATION!)
# Each transaction is scored individually
query = df.withColumn(
    "Risk", 
    predict_fraud("amount", "location", "status", "timestamp")
).writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

print("\n" + "="*60)
print("🔍 TRANSACTION-LEVEL FRAUD DETECTOR RUNNING")
print("="*60)
print("Each transaction is scored individually as it arrives")
print("Press Ctrl+C to stop")
print("="*60 + "\n")

query.awaitTermination()
