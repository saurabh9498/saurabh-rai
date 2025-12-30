# Data Directory

This directory contains transaction data, model files, and sample data for the Real-Time Fraud Detection System.

## Directory Structure

```
data/
├── README.md                    # This file
├── sample/                      # Sample data for testing
│   ├── transactions.json        # Sample transaction records
│   ├── user_profiles.json       # Sample user profiles
│   ├── fraud_labels.json        # Sample fraud labels
│   └── test_batches/            # Pre-batched test data
│       └── batch_100.json
├── models/                      # Trained model files (gitignored)
│   ├── xgboost_v1.pkl
│   ├── neural_net_v1.pt
│   ├── isolation_forest_v1.pkl
│   └── ensemble_config.json
└── features/                    # Feature store snapshots
    └── feature_schema.json
```

## Setting Up Data

### Option 1: Generate Sample Data (Recommended)

Use the provided script to generate synthetic transaction data:

```bash
# From project root
python scripts/generate_sample_data.py

# Generate larger dataset
python scripts/generate_sample_data.py --num-transactions 10000

# Verify files were created
ls -la data/sample/
```

This generates:
- Synthetic transaction records
- User profile histories
- Fraud labels for testing
- Pre-batched data for benchmarking

### Option 2: Use Real Transaction Data

For real-world testing (requires PCI compliance):

1. **Anonymize data**: Remove all PII before using
2. **Format as JSON**:
   ```json
   {
     "transaction_id": "txn_123abc",
     "user_id": "usr_456def",
     "amount": 99.99,
     "currency": "USD",
     "merchant_id": "mrc_789ghi",
     "merchant_category": "retail",
     "timestamp": "2024-01-15T14:30:00Z",
     "device_fingerprint": "fp_abc123",
     "ip_address_hash": "a1b2c3d4...",
     "location": {"lat": 37.7749, "lon": -122.4194}
   }
   ```
3. **Place in data directory**:
   ```bash
   cp your_transactions.json data/transactions.json
   ```

### Option 3: Connect to Streaming Source

For real-time testing, configure Kafka in `.env`:

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_TRANSACTIONS=transactions
KAFKA_CONSUMER_GROUP=fraud-detector
```

## Data Schema

### Transaction Record

| Field | Type | Description |
|-------|------|-------------|
| `transaction_id` | string | Unique transaction identifier |
| `user_id` | string | User identifier |
| `amount` | float | Transaction amount |
| `currency` | string | ISO 4217 currency code |
| `merchant_id` | string | Merchant identifier |
| `merchant_category` | string | MCC category |
| `timestamp` | ISO 8601 | Transaction timestamp |
| `device_fingerprint` | string | Device identifier hash |
| `ip_address_hash` | string | Hashed IP address |
| `location` | object | Lat/lon coordinates |
| `card_present` | boolean | In-person vs online |
| `is_international` | boolean | Cross-border transaction |

### User Profile

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `account_age_days` | int | Days since account creation |
| `avg_transaction_amount` | float | 30-day average |
| `transaction_count_30d` | int | Transactions in last 30 days |
| `usual_locations` | array | Common transaction locations |
| `device_list` | array | Known devices |
| `risk_score_history` | array | Past risk scores |

## Feature Engineering

The system computes these real-time features:

### Velocity Features
- `txn_count_1h`: Transactions in last hour
- `txn_count_24h`: Transactions in last 24 hours
- `amount_sum_1h`: Total amount in last hour
- `amount_sum_24h`: Total amount in last 24 hours

### Deviation Features
- `deviation_from_avg`: Amount vs 30-day average
- `distance_from_usual`: Distance from typical location
- `time_since_last_txn`: Seconds since last transaction

### Risk Indicators
- `is_new_device`: First transaction from this device
- `is_new_merchant`: First transaction with this merchant
- `is_unusual_hour`: Transaction outside normal hours
- `is_high_risk_mcc`: High-risk merchant category

## Model Files

After training, model files are stored in `data/models/`:

```bash
# Train models
python scripts/train_model.py --output data/models/

# Files created:
# - xgboost_v1.pkl (XGBoost classifier)
# - neural_net_v1.pt (PyTorch neural network)
# - isolation_forest_v1.pkl (Anomaly detector)
# - ensemble_config.json (Weight configuration)
```

### Model Versioning

Models are versioned with timestamps:

```
data/models/
├── v1.0.0_20240115/
│   ├── xgboost.pkl
│   ├── neural_net.pt
│   ├── isolation_forest.pkl
│   └── config.json
└── latest -> v1.0.0_20240115/
```

## Feature Store

Redis-based feature store caches user features:

```python
from src.features.feature_store import FeatureStore

store = FeatureStore(redis_url="redis://localhost:6379")

# Get features for user
features = store.get_user_features("usr_123")

# Update features after transaction
store.update_features("usr_123", transaction_data)
```

## Data Privacy & Compliance

⚠️ **Important Security Notes:**

1. **Never commit real transaction data** to version control
2. **Anonymize all test data** - remove names, card numbers, etc.
3. **Hash sensitive fields** - IP addresses, device IDs
4. **Use `.gitignore`** for production data
5. **PCI-DSS compliance** required for card transaction data

### gitignore entries:
```
data/production/
data/transactions.csv
data/*.encrypted
*.pem
*.key
```

## Troubleshooting

### "No training data found"

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Verify
ls data/sample/
```

### "Feature store connection failed"

```bash
# Check Redis is running
redis-cli ping

# Start Redis if needed
docker run -d -p 6379:6379 redis:7
```

### "Model file not found"

```bash
# Train models first
python scripts/train_model.py --output data/models/

# Or download pre-trained (if available)
wget https://your-storage/models/latest.tar.gz
tar -xzf latest.tar.gz -C data/models/
```
