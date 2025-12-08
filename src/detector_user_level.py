from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import joblib

# 1. Setup Spark
spark = SparkSession.builder \
    .appName("FraudDetector") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 2. Define ML Scoring Logic (UDF)
@udf(returnType=StringType())
def get_risk_level(tries, amount, fail_rate, geo_mismatch):
    # Load model (path is relative to where you run spark-submit)
    try:
        model = joblib.load("models/isolation_forest.pkl")
        # Create DataFrame with correct feature names matching training
        features_df = pd.DataFrame({
            'tries': [tries],
            'total_amount': [amount],  # Note: model expects 'total_amount', not 'amount'
            'risk_factor': [fail_rate],  # Note: model expects 'risk_factor', not 'fail_rate'
            'geo_mismatch': [geo_mismatch]
        })
        pred = model.predict(features_df)[0]  # 1 is Normal, -1 is Anomaly
        return "HIGH_RISK" if pred == -1 else "Normal"
    except Exception as e:
        return f"Error: {str(e)}"

# 3. Read Kafka Stream
schema = StructType().add("user_id", StringType()).add("timestamp", StringType()) \
    .add("amount", DoubleType()).add("location", StringType()).add("status", StringType())

df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "payment_attempts").load() \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))

# 4. Create Window Summaries (10 min window)
windowed = df.withWatermark("timestamp", "1 minute") \
    .groupBy(window("timestamp", "10 minutes"), "user_id") \
    .agg(
        count("*").alias("tries"),
        sum("amount").alias("total_amount"),
        mean(when(col("status") == "fail", 1).otherwise(0)).alias("fail_rate"),
        max(when(col("location") != "KR", 1).otherwise(0)).alias("geo_mismatch")
    )

# 5. Apply Model & Output
query = windowed.withColumn("Risk", get_risk_level("tries", "total_amount", "fail_rate", "geo_mismatch")) \
    .writeStream.outputMode("update").format("console").option("truncate", False).start()

query.awaitTermination()