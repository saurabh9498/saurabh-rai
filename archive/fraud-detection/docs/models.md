# Model Documentation

## Overview

The fraud detection system uses a three-model ensemble for robust predictions.

## Ensemble Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE SCORER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   XGBoost   │  │  Neural Net │  │  Isolation Forest   │ │
│  │   (45%)     │  │    (35%)    │  │       (20%)         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                  │
│                 Weighted Average                            │
│                          │                                  │
│                          ▼                                  │
│                    Risk Score                               │
│                     (0 - 1)                                 │
└─────────────────────────────────────────────────────────────┘
```

## Model Details

### 1. XGBoost Classifier

**Purpose**: Capture tabular feature interactions

**Configuration**:
| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 8 |
| learning_rate | 0.05 |
| scale_pos_weight | 10.0 |

**Strengths**:
- Excellent for structured/tabular data
- Handles missing values naturally
- Fast inference (<1ms per sample)
- Built-in feature importance

### 2. Neural Network

**Purpose**: Capture complex non-linear patterns

**Architecture**:
```
Input (17 features)
    │
BatchNorm1d
    │
Linear(17 → 256) + ReLU + Dropout(0.3)
    │
Linear(256 → 128) + ReLU + Dropout(0.3)
    │
Linear(128 → 64) + ReLU + Dropout(0.3)
    │
Linear(64 → 1) + Sigmoid
    │
Output (probability)
```

**Training**:
- Loss: Focal Loss (γ=2.0) for class imbalance
- Optimizer: AdamW with weight decay
- Early stopping on validation loss

### 3. Isolation Forest

**Purpose**: Detect novel anomalies/fraud patterns

**How it works**:
1. Randomly selects features and split points
2. Isolates observations by recursive partitioning
3. Anomalies require fewer splits → shorter path length
4. Score based on average path length

**Configuration**:
| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| contamination | 0.01 |
| max_samples | auto |

## Feature Importance

Top features by importance (XGBoost):

1. `deviation_from_avg` - Amount deviation from 30-day average
2. `txn_count_1h` - Transaction velocity in last hour
3. `amount_sum_24h` - Total spending in 24 hours
4. `is_new_device` - First transaction from device
5. `time_since_last_txn` - Seconds since last transaction

## Decision Thresholds

| Score Range | Decision | Action |
|-------------|----------|--------|
| 0.0 - 0.3 | APPROVE | Auto-approve |
| 0.3 - 0.7 | STEP_UP | Request 2FA |
| 0.7 - 0.9 | REVIEW | Manual queue |
| 0.9 - 1.0 | DECLINE | Auto-decline |

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.994 |
| Precision | 99.7% |
| Recall | 97.3% |
| F1 Score | 98.5% |
| False Positive Rate | 1.8% |

## Model Training

```bash
# Train with synthetic data
python scripts/train_model.py --output models/

# Train with real data
python scripts/train_model.py \
    --data data/transactions.csv \
    --config configs/model_config.yaml \
    --output models/
```

## Model Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --model models/ \
    --data data/test.csv
```

## Online Learning

The system supports incremental model updates:

1. Transactions scored and decisions made
2. Feedback collected (fraud confirmations, false positives)
3. Models updated hourly with new patterns
4. Weights recalibrated based on recent performance

## Model Versioning

Models are versioned with timestamps:
```
models/
├── v1.0.0_20240115/
│   ├── xgboost.pkl
│   ├── neural_net.pt
│   ├── isolation_forest.pkl
│   ├── config.json
│   └── metrics.json
└── latest -> v1.0.0_20240115/
```

## Drift Detection

The system monitors for:

1. **Feature Drift**: PSI score > 0.1 triggers alert
2. **Prediction Drift**: Score distribution changes
3. **Performance Drift**: Precision/recall degradation

See [Monitoring Documentation](deployment.md#monitoring-setup) for details.
