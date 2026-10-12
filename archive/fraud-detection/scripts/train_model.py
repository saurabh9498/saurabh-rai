#!/usr/bin/env python3
"""
Model Training Script

Train the fraud detection ensemble on historical data.

Usage:
    python scripts/train_model.py --data-path data/transactions.parquet --output-dir models/
    
    python scripts/train_model.py \\
        --data-path data/transactions.parquet \\
        --output-dir models/ \\
        --config configs/model_config.yaml \\
        --validate
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import FraudEnsemble, EnsembleConfig
from src.models.xgboost_model import XGBoostConfig
from src.models.neural_net import NeuralNetConfig
from src.models.isolation_forest import IsolationForestConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare training data.
    
    Expected format: Parquet file with features and 'is_fraud' label.
    """
    try:
        import pandas as pd
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        
        # Separate features and labels
        label_col = "is_fraud"
        feature_cols = [c for c in df.columns if c != label_col]
        
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.float32)
        
        logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
        logger.info(f"Fraud rate: {y.mean() * 100:.2f}%")
        
        return X, y
        
    except FileNotFoundError:
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Generating synthetic data for demonstration...")
        return generate_synthetic_data()


def generate_synthetic_data(
    n_samples: int = 100000,
    fraud_rate: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for demonstration."""
    np.random.seed(42)
    
    n_features = 17
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate labels
    y = np.zeros(n_samples, dtype=np.float32)
    n_fraud = int(n_samples * fraud_rate)
    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
    y[fraud_indices] = 1
    
    # Make fraud samples slightly different
    X[fraud_indices] += np.random.randn(n_fraud, n_features) * 0.5
    
    logger.info(f"Generated {n_samples} synthetic samples")
    logger.info(f"Fraud rate: {fraud_rate * 100:.1f}%")
    
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(
    ensemble: FraudEnsemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate model on test set."""
    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    
    logger.info("Evaluating model...")
    
    # Get predictions
    results = ensemble.score_batch(X_test)
    y_pred_proba = np.array([r.risk_score for r in results])
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"FPR:       {metrics['false_positive_rate']:.4f}")
    logger.info("=" * 50)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/transactions.parquet",
        help="Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for training",
    )
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting model training...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config not found, using defaults")
        config = {}
    
    # Load or generate data
    if args.synthetic:
        X, y = generate_synthetic_data()
    else:
        X, y = load_data(args.data_path)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Create ensemble configuration
    ensemble_config = EnsembleConfig(
        xgboost_weight=config.get("ensemble", {}).get("models", {}).get("xgboost", {}).get("weight", 0.45),
        neural_net_weight=config.get("ensemble", {}).get("models", {}).get("neural_net", {}).get("weight", 0.35),
        isolation_forest_weight=config.get("ensemble", {}).get("models", {}).get("isolation_forest", {}).get("weight", 0.20),
    )
    
    # Feature names (for explainability)
    feature_names = [
        "txn_count_1h", "txn_count_6h", "txn_count_24h", "txn_count_7d",
        "amount_sum_1h", "amount_sum_24h", "amount_avg_30d", "amount_std_30d",
        "time_since_last_txn", "unique_merchants_24h", "unique_channels_24h",
        "is_first_transaction", "is_new_merchant", "is_new_device",
        "deviation_from_avg", "merchant_risk_score", "device_risk_score",
    ]
    
    # Create and train ensemble
    ensemble = FraudEnsemble(
        config=ensemble_config,
        feature_names=feature_names,
    )
    
    training_metrics = ensemble.train(X_train, y_train, X_val, y_val)
    
    # Evaluate if requested
    if args.validate:
        eval_metrics = evaluate_model(ensemble, X_test, y_test)
        training_metrics["evaluation"] = eval_metrics
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"ensemble_{timestamp}"
    ensemble.save(str(model_path))
    
    # Save training metrics
    import json
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
