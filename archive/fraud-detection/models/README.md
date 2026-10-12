# Trained Models Directory

This directory contains trained model artifacts for the fraud detection ensemble.

## Directory Structure

```
models/
├── README.md                   # This file
├── xgboost/
│   ├── model_v1.0.0.pkl       # XGBoost classifier
│   ├── model_v1.0.0_meta.json # Model metadata
│   └── feature_importance.json # Feature importance scores
├── neural_net/
│   ├── model_v1.0.0.pt        # PyTorch model weights
│   ├── model_v1.0.0_meta.json # Model metadata
│   └── architecture.json       # Network architecture definition
├── isolation_forest/
│   ├── model_v1.0.0.pkl       # Isolation Forest model
│   └── model_v1.0.0_meta.json # Model metadata
└── ensemble/
    ├── weights_v1.0.0.json    # Ensemble weights configuration
    └── thresholds_v1.0.0.json # Decision thresholds
```

## Model Versioning

Models follow semantic versioning: `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking changes in features or output format
- **MINOR**: New features or significant retraining
- **PATCH**: Bug fixes or minor retraining

## Model Metadata Schema

Each model includes a `*_meta.json` file with:

```json
{
  "model_name": "xgboost_fraud_detector",
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "training_data": {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "num_samples": 5000000,
    "fraud_rate": 0.023
  },
  "features": {
    "count": 45,
    "categories": ["velocity", "deviation", "risk_indicators"]
  },
  "performance": {
    "auc_roc": 0.9845,
    "precision_at_95_recall": 0.82,
    "f1_score": 0.87
  },
  "hyperparameters": {
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500
  }
}
```

## Training Models

```bash
# Train all models
make train

# Train specific model
python scripts/train_model.py --model xgboost --data data/sample/

# Train with custom hyperparameters
python scripts/train_model.py \
    --model xgboost \
    --data data/sample/ \
    --config configs/model_config.yaml
```

## Loading Models

```python
from src.models.ensemble import FraudEnsemble

# Load latest models
ensemble = FraudEnsemble.from_directory("models/")

# Load specific version
ensemble = FraudEnsemble.from_directory("models/", version="1.0.0")

# Score a transaction
result = ensemble.score(transaction, features)
print(f"Risk Score: {result.risk_score}")
print(f"Decision: {result.decision}")
```

## Model Registry Integration

For production, models should be registered in MLflow:

```python
import mlflow

# Log model to MLflow
with mlflow.start_run():
    mlflow.xgboost.log_model(model, "xgboost_fraud_detector")
    mlflow.log_params(hyperparameters)
    mlflow.log_metrics(metrics)
```

## Model Performance Requirements

| Metric | Minimum | Target | Current |
|--------|---------|--------|---------|
| AUC-ROC | 0.95 | 0.98 | 0.9845 |
| Precision @ 95% Recall | 0.70 | 0.85 | 0.82 |
| False Positive Rate | < 3% | < 2% | 1.8% |
| Inference Latency (P99) | < 50ms | < 20ms | 8ms |

## Model Retraining

Models are retrained on a scheduled basis:
- **XGBoost**: Weekly (Sundays at 2 AM UTC)
- **Neural Network**: Bi-weekly
- **Isolation Forest**: Monthly
- **Ensemble Weights**: After any component model update

## Git LFS

Large model files (`.pkl`, `.pt`) should be tracked with Git LFS:

```bash
git lfs track "*.pkl"
git lfs track "*.pt"
```

## Security Notes

⚠️ **Important**: 
- Never commit production models to public repositories
- Model files may contain sensitive feature engineering logic
- Use environment variables for model paths in production
