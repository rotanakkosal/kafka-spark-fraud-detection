import time
import json
import random
from kafka import KafkaProducer
from datetime import datetime

TOPIC = "payment_attempts"
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print(f"✅ Sending data to {TOPIC}...")

def generate_event(fraud=False):
    # If fraud: High amount ($500-5000), foreign country, high fail chance
    if fraud:
        return {
            "user_id": f"user_{random.randint(1,5)}", # Attack specific users
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(random.uniform(500, 5000), 2),
            "location": random.choice(["CN", "RU", "US"]),
            "status": random.choice(["fail", "fail", "success"])
        }
    # If normal: Low amount, KR location, success
    else:
        return {
            "user_id": f"user_{random.randint(1,20)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(random.uniform(10, 100), 2),
            "location": "KR",
            "status": "success"
        }

while True:
    # 90% chance normal, 10% chance fraud attack
    is_attack = random.random() < 0.1
    
    if is_attack:
        print("⚠️  SENDING FRAUD BURST!")
        for _ in range(5):
            producer.send(TOPIC, generate_event(fraud=True))
    else:
        print(".. sending normal event")
        producer.send(TOPIC, generate_event(fraud=False))
        
    time.sleep(1)