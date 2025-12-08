# Real-Time Fraud Detection with Kafka, Spark, and Machine Learning

## Project Overview

This project implements an end-to-end streaming pipeline to detect fraudulent financial transactions in real-time. It leverages a modern Big Data stack, including **Apache Kafka** for event streaming, **Apache Spark** for distributed stream processing, and **Scikit-learn** for machine learning.

The core of this project is not just building a model, but demonstrating the critical, iterative process of diagnosing flaws, fixing them, and validating a robust, trustworthy solution suitable for a real-world scenario.

### Core Technologies
*   **Streaming:** Apache Kafka, Docker
*   **Processing:** Apache Spark (Structured Streaming)
*   **Machine Learning:** Scikit-learn (Random Forest)
*   **Data Simulation:** Python (Kafka Producer)
*   **Dataset:** PaySim Synthetic Financial Data

---

## The Story: A Journey from a Flawed Model to a Robust Solution

This project documents a realistic data science workflow, moving from a simplistic first attempt to a sophisticated, validated final model.

### Chapter 1: The Initial Flawed Approach - The User-Level Model

Our first attempt focused on aggregating user behavior over time, a common approach in behavioral analytics.

*   **Methodology:**
    *   **Model:** `IsolationForest` (Unsupervised Anomaly Detection).
    *   **Granularity:** User-Level. All transactions for a single user in the dataset were grouped to create a single profile.
    *   **Features:** `tries` (transaction count), `total_amount`, `fail_rate`, and a simulated `geo_mismatch`.

*   **Results:** The performance was extremely poor.
    *   **Precision: 4%** - 96% of fraud alerts were false alarms.
    *   **Recall: 29%** - The model missed 71% of all actual fraud.

*   **Conclusion:** This approach failed because user-level aggregation loses the critical, transaction-specific details where fraud patterns are most evident. The features were too generic to capture sophisticated fraud signals.

### Chapter 2: The Investigation - The "Illusion of Perfection"

The second attempt shifted to a transaction-level model, which immediately yielded suspiciously perfect results.

*   **Methodology:**
    *   **Model:** `RandomForestClassifier` (Supervised Learning).
    *   **Granularity:** Transaction-Level. Each transaction was evaluated individually.
    *   **Initial Results:** The model achieved **99.94% Precision and 99.28% Recall**.

*   **The Red Flag:** Such near-perfect scores in fraud detection are almost always a sign of a problem, not success. This indicated a high probability of **data leakage**, where the model "cheats" by using information it would not have in a real-world prediction scenario.

*   **The Diagnosis:**
    1.  **Overfitting Check:** We first confirmed the model wasn't simply memorizing the training data. A diagnostic script (`check_overfitting.py`) showed the model generalized well to other samples from the *same dataset*, proving the issue was deeper than simple overfitting.
    2.  **Feature Importance Analysis:** A second script (`find_leak.py`) revealed the source of the problem. The model's most important features were:
        *   `balance_change_orig` (Old Balance - New Balance - Amount)
        *   `newbalance_orig` (The sender's balance *after* the transaction)
        *   `balance_to_zero` (A flag for when the sender's balance becomes zero)

    **The Leak Explained:** In a live system, you must decide to approve or deny a transaction *before* it is processed. At that moment, you do not know what the `newbalance_orig` will be. These features were leaking information from the future to the model, giving it an unfair and unrealistic advantage.

### Chapter 3: Building a Robust Solution - A Realistic Approach

Armed with the knowledge of the data leak, we built a third and final model designed for a real-world scenario.

*   **Methodology:**
    *   **Feature Engineering:** We created a feature set using **only pre-transaction data**—information that would be available at the moment of decision. Leaky features were removed.
        *   **Final Features:** `step`, `amount`, `log_amount`, `oldbalance_orig`, `oldbalance_dest`, `balance_ratio`, `type_TRANSFER`, `type_CASH_OUT`.
    *   **Model Regularization:** The `RandomForestClassifier` was trained with regularization parameters (`max_depth=10`, `min_samples_leaf=10`) to prevent overfitting and encourage it to learn more general patterns.
    *   **Best Practices:** The data was properly split into training (80%) and testing (20%) sets to ensure a fair evaluation on completely unseen data.

### Chapter 4: Trustworthy Results - A Model Ready for Production

The final, robust model produced results that are not only excellent but also realistic and trustworthy.

*   **Final Performance (on unseen test data):**
    *   **Precision: 85.63%** - When an alert is raised, it's correct 86% of the time. This is a strong, actionable signal with a low rate of false alarms.
    *   **Recall: 99.34%** - The model successfully catches 99.3% of all fraudulent transactions, representing extremely low risk exposure.
    *   **F1-Score: 0.9198** - A high F1-score indicates an excellent balance between precision and recall.

*   **Final Overfitting Diagnosis:** A final diagnostic (`final_diagnostic.py`) confirmed that the performance gap between the training set and the test set was near zero, proving the model generalizes perfectly and is not overfit.

    ```
    Performance Gap:
       - Train F1-Score: 0.9156
       - Test F1-Score:  0.9198
       --------------------
       - Gap:            -0.0042  (Excellent!)
    ```

*   **Conclusion:** This iterative process demonstrates a mature data science workflow. By identifying and fixing data leakage, we transformed a useless model into a high-performing, reliable, and production-ready fraud detection engine.

---

## System Architecture

The real-time pipeline consists of four main components:

```
[Interactive Producer] -> [Kafka Topic] -> [Spark Streaming Detector] -> [Console Output]
       (Python)         (payment_attempts)     (PySpark, Scikit-learn)
```

1.  **Interactive Producer (`interactive_producer.py`):** A Python script that allows a user to manually input transaction data (amount, type, balances) and sends it to Kafka as a JSON message.
2.  **Kafka:** A Dockerized message broker that ingests the streaming transaction data into the `payment_attempts` topic.
3.  **Spark Streaming Detector (`detector_transaction_level.py`):** A PySpark application that:
    *   Connects to the Kafka topic.
    *   Reads transaction data in real-time micro-batches.
    *   Uses a User-Defined Function (UDF) to apply the trained Scikit-learn model and scaler to each individual transaction.
    *   Generates a prediction ("HIGH_RISK" or "Normal").
4.  **Console Output:** The final predictions are printed to the console in real-time.

---

## How to Run the Project

Follow these steps to set up and run the entire pipeline.

### 1. Prerequisites
*   Docker & Docker Compose
*   Python 3.9+ and a virtual environment
*   Java (for Spark)
*   Apache Spark installed and configured

### 2. Initial Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd realtime_fraud_detection

# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Start Infrastructure
Start the Kafka and Zookeeper containers.
```bash
docker-compose up -d
```

### 4. Train the Robust Model
Run the training script to create `fraud_model_robust.pkl` and `scaler_robust.pkl`.
```bash
python src/model_trainer/trainer_robust.py
```

### 5. Validate the Model (Optional)
Run the validation script to confirm the model's performance on unseen data.
```bash
python src/validate_model/validate_robust.py
```

### 6. Run the Real-Time System

**In Terminal 1 - Start the Spark Detector:**
```bash
spark-submit src/detector_transaction_level.py
```
Wait for the detector to initialize and start listening for messages.

**In Terminal 2 - Start the Interactive Producer:**
```bash
python src/interactive_producer.py
```
Follow the prompts to send test transactions. You will see the predictions appear in Terminal 1.

---

## Conclusion and Future Work

This project successfully demonstrates the end-to-end process of building, diagnosing, and refining a machine learning model for a real-time big data application. The key takeaway is the critical importance of moving beyond superficial metrics like accuracy to deeply investigate model behavior, identify issues like data leakage, and produce a final result that is both high-performing and trustworthy.

**Future Work:**
*   **Test on Real Data:** The current model is an expert on the PaySim dataset. The next logical step would be to test its performance on a real-world dataset, like the IEEE-CIS Fraud Detection dataset, to evaluate its generalization to more complex and noisy data.
*   **Deploy to a Cluster:** For a true production scenario, the Spark application would be deployed to a distributed cluster (e.g., AWS EMR, Databricks) to handle massive transaction volumes.
*   **Develop a Real-Time Dashboard:** Instead of printing to the console, the output stream could be fed into a real-time dashboarding tool like Grafana or Kibana for visualization.
