#!/usr/bin/env python3
"""
Sample Data Generator for Real-Time Fraud Detection System

Generates synthetic transaction data for development and testing:
- Transaction records with realistic patterns
- User profiles with history
- Fraud labels for training/evaluation
- Pre-batched data for performance testing

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --num-transactions 10000
    python scripts/generate_sample_data.py --fraud-rate 0.02
    python scripts/generate_sample_data.py --force
"""

import argparse
import json
import os
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import uuid


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


# Merchant categories with risk levels
MERCHANT_CATEGORIES = {
    "grocery": {"risk": 0.01, "avg_amount": 75},
    "restaurant": {"risk": 0.02, "avg_amount": 45},
    "gas_station": {"risk": 0.03, "avg_amount": 55},
    "retail": {"risk": 0.02, "avg_amount": 120},
    "online_retail": {"risk": 0.05, "avg_amount": 85},
    "travel": {"risk": 0.04, "avg_amount": 450},
    "entertainment": {"risk": 0.03, "avg_amount": 35},
    "utilities": {"risk": 0.01, "avg_amount": 150},
    "healthcare": {"risk": 0.02, "avg_amount": 200},
    "electronics": {"risk": 0.06, "avg_amount": 350},
    "gambling": {"risk": 0.15, "avg_amount": 100},
    "crypto_exchange": {"risk": 0.20, "avg_amount": 500},
    "wire_transfer": {"risk": 0.12, "avg_amount": 1000},
}

# Geographic locations (lat, lon, city)
LOCATIONS = [
    (37.7749, -122.4194, "San Francisco"),
    (40.7128, -74.0060, "New York"),
    (34.0522, -118.2437, "Los Angeles"),
    (41.8781, -87.6298, "Chicago"),
    (29.7604, -95.3698, "Houston"),
    (33.4484, -112.0740, "Phoenix"),
    (39.7392, -104.9903, "Denver"),
    (47.6062, -122.3321, "Seattle"),
    (25.7617, -80.1918, "Miami"),
    (42.3601, -71.0589, "Boston"),
]

# Fraud patterns
FRAUD_PATTERNS = [
    "card_testing",      # Small amounts to test stolen card
    "account_takeover",  # Unusual location/device
    "bust_out",          # Rapid high-value transactions
    "friendly_fraud",    # Legitimate user disputes
    "synthetic_id",      # Fake identity
]


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return f"usr_{uuid.uuid4().hex[:12]}"


def generate_transaction_id() -> str:
    """Generate a unique transaction ID."""
    return f"txn_{uuid.uuid4().hex[:16]}"


def generate_merchant_id(category: str) -> str:
    """Generate a merchant ID."""
    return f"mrc_{category[:3]}_{uuid.uuid4().hex[:8]}"


def hash_value(value: str) -> str:
    """Create a hash of a value (for IPs, devices, etc.)."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def generate_device_fingerprint() -> str:
    """Generate a device fingerprint."""
    return f"fp_{uuid.uuid4().hex[:12]}"


def generate_ip_hash() -> str:
    """Generate a hashed IP address."""
    ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
    return hash_value(ip)


def generate_user_profile(user_id: str, num_transactions: int = 50) -> Dict[str, Any]:
    """Generate a user profile with transaction history."""
    home_location = random.choice(LOCATIONS)
    account_age = random.randint(30, 1825)  # 1 month to 5 years
    
    # Generate typical transaction patterns
    preferred_categories = random.sample(list(MERCHANT_CATEGORIES.keys()), k=random.randint(3, 6))
    
    avg_amount = sum(MERCHANT_CATEGORIES[c]["avg_amount"] for c in preferred_categories) / len(preferred_categories)
    avg_amount *= random.uniform(0.7, 1.5)  # Add some variation
    
    return {
        "user_id": user_id,
        "account_age_days": account_age,
        "avg_transaction_amount": round(avg_amount, 2),
        "transaction_count_30d": random.randint(10, 100),
        "home_location": {
            "lat": home_location[0],
            "lon": home_location[1],
            "city": home_location[2]
        },
        "usual_locations": [
            {"lat": home_location[0] + random.uniform(-0.1, 0.1),
             "lon": home_location[1] + random.uniform(-0.1, 0.1)}
            for _ in range(random.randint(2, 5))
        ],
        "preferred_categories": preferred_categories,
        "devices": [generate_device_fingerprint() for _ in range(random.randint(1, 3))],
        "risk_score_history": [random.uniform(0.01, 0.15) for _ in range(10)],
        "created_at": (datetime.now() - timedelta(days=account_age)).isoformat()
    }


def generate_legitimate_transaction(
    user_id: str,
    user_profile: Dict[str, Any],
    timestamp: datetime
) -> Dict[str, Any]:
    """Generate a legitimate transaction based on user profile."""
    # Pick a category the user typically uses
    category = random.choice(user_profile["preferred_categories"])
    category_info = MERCHANT_CATEGORIES[category]
    
    # Amount near user's average with some variation
    base_amount = user_profile["avg_transaction_amount"]
    amount = max(1, base_amount * random.uniform(0.3, 2.5))
    
    # Location near user's home
    home = user_profile["home_location"]
    location = {
        "lat": home["lat"] + random.uniform(-0.05, 0.05),
        "lon": home["lon"] + random.uniform(-0.05, 0.05)
    }
    
    # Use known device
    device = random.choice(user_profile["devices"])
    
    return {
        "transaction_id": generate_transaction_id(),
        "user_id": user_id,
        "amount": round(amount, 2),
        "currency": "USD",
        "merchant_id": generate_merchant_id(category),
        "merchant_category": category,
        "timestamp": timestamp.isoformat(),
        "device_fingerprint": device,
        "ip_address_hash": generate_ip_hash(),
        "location": location,
        "card_present": random.random() > 0.3,  # 70% in-person
        "is_international": False,
        "is_fraud": False,
        "fraud_pattern": None
    }


def generate_fraudulent_transaction(
    user_id: str,
    user_profile: Dict[str, Any],
    timestamp: datetime,
    pattern: str = None
) -> Dict[str, Any]:
    """Generate a fraudulent transaction."""
    if pattern is None:
        pattern = random.choice(FRAUD_PATTERNS)
    
    txn = generate_legitimate_transaction(user_id, user_profile, timestamp)
    txn["is_fraud"] = True
    txn["fraud_pattern"] = pattern
    
    if pattern == "card_testing":
        # Small amounts to test if card works
        txn["amount"] = round(random.uniform(0.50, 5.00), 2)
        txn["merchant_category"] = "online_retail"
        txn["card_present"] = False
        
    elif pattern == "account_takeover":
        # Unusual location and new device
        unusual_location = random.choice([l for l in LOCATIONS 
                                          if l[2] != user_profile["home_location"]["city"]])
        txn["location"] = {"lat": unusual_location[0], "lon": unusual_location[1]}
        txn["device_fingerprint"] = generate_device_fingerprint()  # New device
        txn["amount"] = round(user_profile["avg_transaction_amount"] * random.uniform(2, 5), 2)
        
    elif pattern == "bust_out":
        # High value transaction
        txn["amount"] = round(random.uniform(500, 5000), 2)
        txn["merchant_category"] = random.choice(["electronics", "wire_transfer", "crypto_exchange"])
        
    elif pattern == "friendly_fraud":
        # Looks legitimate but will be disputed
        pass  # Keep it looking normal
        
    elif pattern == "synthetic_id":
        # New account, unusual patterns
        txn["device_fingerprint"] = generate_device_fingerprint()
        txn["merchant_category"] = random.choice(["crypto_exchange", "gambling", "wire_transfer"])
        txn["amount"] = round(random.uniform(200, 2000), 2)
    
    return txn


def generate_transactions(
    num_transactions: int = 1000,
    num_users: int = 100,
    fraud_rate: float = 0.02,
    days_back: int = 30
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate synthetic transaction data.
    
    Returns:
        Tuple of (transactions, user_profiles, fraud_labels)
    """
    # Generate users
    users = {}
    for _ in range(num_users):
        user_id = generate_user_id()
        users[user_id] = generate_user_profile(user_id)
    
    # Generate transactions
    transactions = []
    fraud_labels = []
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    for i in range(num_transactions):
        # Pick a random user
        user_id = random.choice(list(users.keys()))
        user_profile = users[user_id]
        
        # Random timestamp
        timestamp = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )
        
        # Decide if fraudulent
        is_fraud = random.random() < fraud_rate
        
        if is_fraud:
            txn = generate_fraudulent_transaction(user_id, user_profile, timestamp)
        else:
            txn = generate_legitimate_transaction(user_id, user_profile, timestamp)
        
        transactions.append(txn)
        fraud_labels.append({
            "transaction_id": txn["transaction_id"],
            "is_fraud": txn["is_fraud"],
            "fraud_pattern": txn.get("fraud_pattern"),
            "confidence": 1.0 if is_fraud else random.uniform(0.95, 1.0)
        })
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x["timestamp"])
    
    return transactions, list(users.values()), fraud_labels


def create_feature_schema() -> Dict[str, Any]:
    """Create feature schema documentation."""
    return {
        "version": "1.0.0",
        "features": {
            "velocity_features": [
                {"name": "txn_count_1h", "type": "int", "description": "Transactions in last hour"},
                {"name": "txn_count_24h", "type": "int", "description": "Transactions in last 24 hours"},
                {"name": "amount_sum_1h", "type": "float", "description": "Total amount in last hour"},
                {"name": "amount_sum_24h", "type": "float", "description": "Total amount in last 24 hours"},
            ],
            "deviation_features": [
                {"name": "deviation_from_avg", "type": "float", "description": "Amount deviation from 30-day average"},
                {"name": "distance_from_usual", "type": "float", "description": "Distance from typical location (km)"},
                {"name": "time_since_last_txn", "type": "int", "description": "Seconds since last transaction"},
            ],
            "risk_indicators": [
                {"name": "is_new_device", "type": "bool", "description": "First transaction from this device"},
                {"name": "is_new_merchant", "type": "bool", "description": "First transaction with this merchant"},
                {"name": "is_unusual_hour", "type": "bool", "description": "Transaction outside normal hours"},
                {"name": "is_high_risk_mcc", "type": "bool", "description": "High-risk merchant category"},
            ],
            "model_inputs": [
                {"name": "amount_normalized", "type": "float", "description": "Log-normalized amount"},
                {"name": "hour_of_day", "type": "int", "description": "Hour (0-23)"},
                {"name": "day_of_week", "type": "int", "description": "Day (0-6)"},
                {"name": "merchant_risk_score", "type": "float", "description": "Historical merchant risk"},
            ]
        },
        "target": {
            "name": "is_fraud",
            "type": "bool",
            "description": "Whether transaction is fraudulent"
        }
    }


def write_json(path: Path, data: Any) -> None:
    """Write JSON data to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Created: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample data for Fraud Detection System"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: project data/)"
    )
    parser.add_argument(
        "--num-transactions", "-n", type=int, default=1000,
        help="Number of transactions to generate"
    )
    parser.add_argument(
        "--num-users", "-u", type=int, default=100,
        help="Number of unique users"
    )
    parser.add_argument(
        "--fraud-rate", "-f", type=float, default=0.02,
        help="Fraud rate (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--days-back", "-d", type=int, default=30,
        help="Days of history to generate"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files"
    )
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        data_dir = args.output
    else:
        data_dir = get_project_root() / "data"
    
    sample_dir = data_dir / "sample"
    features_dir = data_dir / "features"
    
    print("=" * 60)
    print("Fraud Detection System - Sample Data Generator")
    print("=" * 60)
    print(f"\nOutput directory: {data_dir}")
    print(f"Generating {args.num_transactions} transactions for {args.num_users} users")
    print(f"Fraud rate: {args.fraud_rate:.1%}")
    print()
    
    # Generate transactions
    print("📊 Generating Transaction Data...")
    print("-" * 40)
    
    transactions, user_profiles, fraud_labels = generate_transactions(
        num_transactions=args.num_transactions,
        num_users=args.num_users,
        fraud_rate=args.fraud_rate,
        days_back=args.days_back
    )
    
    # Calculate statistics
    fraud_count = sum(1 for t in transactions if t["is_fraud"])
    legitimate_count = len(transactions) - fraud_count
    
    print(f"  Generated {len(transactions)} transactions")
    print(f"  - Legitimate: {legitimate_count} ({legitimate_count/len(transactions):.1%})")
    print(f"  - Fraudulent: {fraud_count} ({fraud_count/len(transactions):.1%})")
    
    # Write files
    print("\n💾 Saving Data Files...")
    print("-" * 40)
    
    write_json(sample_dir / "transactions.json", transactions)
    write_json(sample_dir / "user_profiles.json", user_profiles)
    write_json(sample_dir / "fraud_labels.json", fraud_labels)
    
    # Create batched data for performance testing
    batch_size = 100
    batches = [transactions[i:i+batch_size] for i in range(0, len(transactions), batch_size)]
    (sample_dir / "test_batches").mkdir(parents=True, exist_ok=True)
    
    for i, batch in enumerate(batches[:5]):  # Save first 5 batches
        write_json(sample_dir / "test_batches" / f"batch_{i}.json", batch)
    
    # Create feature schema
    print("\n📋 Creating Feature Schema...")
    print("-" * 40)
    
    write_json(features_dir / "feature_schema.json", create_feature_schema())
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Sample data generation complete!")
    print("=" * 60)
    print(f"""
Files created:
  Sample Data:
    - transactions.json ({len(transactions)} records)
    - user_profiles.json ({len(user_profiles)} users)
    - fraud_labels.json ({len(fraud_labels)} labels)
    - test_batches/ ({min(5, len(batches))} batches of {batch_size})

  Feature Schema:
    - feature_schema.json

Statistics:
    - Total transactions: {len(transactions)}
    - Legitimate: {legitimate_count} ({legitimate_count/len(transactions):.1%})
    - Fraudulent: {fraud_count} ({fraud_count/len(transactions):.1%})
    - Unique users: {len(user_profiles)}
    - Date range: {args.days_back} days

Next steps:
  1. Train models: python scripts/train_model.py --data data/sample/
  2. Evaluate: python scripts/evaluate.py --data data/sample/
  3. Run API: uvicorn src.api.main:app --reload
""")


if __name__ == "__main__":
    main()
