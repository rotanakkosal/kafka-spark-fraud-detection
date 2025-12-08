import json
from kafka import KafkaProducer
from datetime import datetime
import random

TOPIC = "payment_attempts"
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("="*60)
print("🔍 ROBUST FRAUD DETECTION TESTER")
print("="*60)
print("\nEnter transaction details to test the robust model.")
print("Type 'quit' to exit.\n")

def send_event(user_id, amount, oldbalance_orig, oldbalance_dest, transaction_type):
    """Send a single event to Kafka with all required fields."""
    event = {
        "user_id": user_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount": float(amount),
        "oldbalance_orig": float(oldbalance_orig),
        "oldbalance_dest": float(oldbalance_dest),
        "type": transaction_type.upper()
    }
    producer.send(TOPIC, event)
    producer.flush()
    print(f"✅ Sent: {event}")
    return event

def test_scenario():
    """Run predefined test scenarios for the robust model."""
    print("\n📋 Test Scenarios:")
    print("1. Normal PAYMENT (low risk)")
    print("2. Fraudulent TRANSFER (drains account, should be HIGH_RISK)")
    print("3. Fraudulent CASH_OUT (follows transfer, should be HIGH_RISK)")
    print("4. High-value but legitimate TRANSFER (should be Normal)")
    print("5. Custom input")
    
    choice = input("\nSelect scenario (1-5): ").strip()
    
    if choice == "1":
        # Normal PAYMENT is not a fraud type in our model
        send_event("user_normal_1", 50.0, 10000.0, 2000.0, "PAYMENT")
        print("→ Expected: Normal (PAYMENT is not a fraud type in the model)")
        
    elif choice == "2":
        # Classic fraud: TRANSFER draining the account
        amount = 50000.0
        old_balance = 50000.0
        send_event("user_fraud_1", amount, old_balance, 0.0, "TRANSFER")
        print("→ Expected: HIGH_RISK (TRANSFER, balance_ratio is 1.0)")

    elif choice == "3":
        # The CASH_OUT that often follows a fraudulent TRANSFER
        amount = 49999.0
        old_balance = 49999.0
        send_event("user_fraud_1", amount, old_balance, 0.0, "CASH_OUT")
        print("→ Expected: HIGH_RISK (CASH_OUT, balance_ratio is ~1.0)")
        
    elif choice == "4":
        # A large but legitimate transfer where the balance ratio is not 1.0
        amount = 100000.0
        old_balance = 500000.0
        send_event("user_corp_1", amount, old_balance, 100000.0, "TRANSFER")
        print("→ Expected: Normal (Large amount, but balance_ratio is low)")

    elif choice == "5":
        # Custom input
        user_id = input("User ID: ").strip()
        transaction_type = input("Type (TRANSFER or CASH_OUT): ").strip().upper()
        amount = float(input("Amount: ").strip())
        oldbalance_orig = float(input("Sender Old Balance: ").strip())
        oldbalance_dest = float(input("Recipient Old Balance: ").strip())
        send_event(user_id, amount, oldbalance_orig, oldbalance_dest, transaction_type)
        
    else:
        print("Invalid choice!")

def main():
    while True:
        action = input("\nPress 't' for test scenarios, 'c' for custom, or 'q' to quit: ").strip().lower()
        
        if action == "q":
            print("👋 Goodbye!")
            break
        elif action == "t":
            test_scenario()
        elif action == "c":
            user_id = input("User ID: ").strip()
            transaction_type = input("Type (TRANSFER or CASH_OUT): ").strip().upper()
            amount = float(input("Amount: ").strip())
            oldbalance_orig = float(input("Sender Old Balance: ").strip())
            oldbalance_dest = float(input("Recipient Old Balance: ").strip())
            send_event(user_id, amount, oldbalance_orig, oldbalance_dest, transaction_type)
        else:
            print("Invalid option.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    finally:
        producer.close()
