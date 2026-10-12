#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluate fraud detection model performance with comprehensive metrics.

Usage:
    python scripts/evaluate.py --model-path models/ensemble_latest --data-path data/test.parquet
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_data(data_path: str):
    """Load test data."""
    try:
        import pandas as pd
        df = pd.read_parquet(data_path)
        X = df.drop("is_fraud", axis=1).values.astype(np.float32)
        y = df["is_fraud"].values.astype(np.float32)
        return X, y
    except FileNotFoundError:
        logger.info("Generating synthetic test data...")
        np.random.seed(123)
        n_samples = 10000
        X = np.random.randn(n_samples, 17).astype(np.float32)
        y = np.zeros(n_samples, dtype=np.float32)
        y[np.random.choice(n_samples, int(n_samples * 0.01))] = 1
        return X, y


def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_curve,
        average_precision_score,
        roc_curve,
    )
    
    metrics = {}
    
    # AUC metrics
    metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba))
    metrics["auc_pr"] = float(average_precision_score(y_true, y_pred_proba))
    
    # Threshold-based metrics at different operating points
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics[f"threshold_{thresh}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "fpr": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
    
    # Score distribution
    metrics["score_distribution"] = {
        "mean": float(np.mean(y_pred_proba)),
        "std": float(np.std(y_pred_proba)),
        "median": float(np.median(y_pred_proba)),
        "p90": float(np.percentile(y_pred_proba, 90)),
        "p95": float(np.percentile(y_pred_proba, 95)),
        "p99": float(np.percentile(y_pred_proba, 99)),
    }
    
    # Score distribution by class
    metrics["fraud_scores"] = {
        "mean": float(np.mean(y_pred_proba[y_true == 1])),
        "std": float(np.std(y_pred_proba[y_true == 1])) if np.sum(y_true == 1) > 1 else 0,
    }
    metrics["legit_scores"] = {
        "mean": float(np.mean(y_pred_proba[y_true == 0])),
        "std": float(np.std(y_pred_proba[y_true == 0])),
    }
    
    return metrics


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    amounts: np.ndarray = None,
    review_cost: float = 10.0,
    fraud_loss_rate: float = 1.0,
) -> Dict[str, Any]:
    """Calculate business-oriented metrics."""
    
    if amounts is None:
        amounts = np.random.uniform(50, 500, len(y_true))
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    business_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        # Fraud caught
        fraud_caught = np.sum((y_pred == 1) & (y_true == 1))
        fraud_missed = np.sum((y_pred == 0) & (y_true == 1))
        
        # False positives (unnecessary reviews)
        false_positives = np.sum((y_pred == 1) & (y_true == 0))
        
        # Financial impact
        fraud_loss = np.sum(amounts[(y_pred == 0) & (y_true == 1)]) * fraud_loss_rate
        review_cost_total = false_positives * review_cost
        fraud_saved = np.sum(amounts[(y_pred == 1) & (y_true == 1)])
        
        net_savings = fraud_saved - review_cost_total
        
        business_metrics[f"threshold_{thresh}"] = {
            "fraud_caught": int(fraud_caught),
            "fraud_missed": int(fraud_missed),
            "false_positives": int(false_positives),
            "fraud_loss_usd": float(fraud_loss),
            "review_cost_usd": float(review_cost_total),
            "fraud_saved_usd": float(fraud_saved),
            "net_savings_usd": float(net_savings),
        }
    
    return business_metrics


def print_report(metrics: Dict[str, Any], business: Dict[str, Any]):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("                    FRAUD DETECTION MODEL EVALUATION")
    print("=" * 70)
    
    print(f"\n{'Overall Metrics':=^50}")
    print(f"  AUC-ROC:              {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:               {metrics['auc_pr']:.4f}")
    
    print(f"\n{'Score Distribution':=^50}")
    dist = metrics["score_distribution"]
    print(f"  Mean:    {dist['mean']:.4f}")
    print(f"  Median:  {dist['median']:.4f}")
    print(f"  P90:     {dist['p90']:.4f}")
    print(f"  P99:     {dist['p99']:.4f}")
    
    print(f"\n{'Threshold Analysis':=^50}")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'FPR':<12}")
    print("-" * 50)
    
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        m = metrics[f"threshold_{thresh}"]
        print(f"  {thresh:<12.1f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['fpr']:<12.4f}")
    
    print(f"\n{'Business Impact (per 10K transactions)':=^50}")
    print(f"  {'Threshold':<12} {'Fraud Caught':<15} {'FP Reviews':<15} {'Net Savings':<15}")
    print("-" * 60)
    
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        b = business[f"threshold_{thresh}"]
        print(f"  {thresh:<12.1f} {b['fraud_caught']:<15} {b['false_positives']:<15} ${b['net_savings_usd']:<14.2f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--model-path", type=str, default="models/ensemble_latest")
    parser.add_argument("--data-path", type=str, default="data/test.parquet")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()
    
    logger.info("Loading test data...")
    X, y = load_test_data(args.data_path)
    logger.info(f"Loaded {len(X)} samples, fraud rate: {y.mean()*100:.2f}%")
    
    # Try to load model, fallback to mock
    try:
        from src.models.ensemble import FraudEnsemble
        ensemble = FraudEnsemble.load(args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.info("Using mock ensemble for demonstration...")
        
        from src.models.ensemble import FraudEnsemble
        from src.models.xgboost_model import MockXGBoostModel
        from src.models.neural_net import MockNeuralNetModel
        from src.models.isolation_forest import MockIsolationForestModel
        
        ensemble = FraudEnsemble()
        ensemble.xgboost = MockXGBoostModel()
        ensemble.neural_net = MockNeuralNetModel()
        ensemble.isolation_forest = MockIsolationForestModel()
        ensemble._is_trained = True
    
    logger.info("Scoring transactions...")
    results = ensemble.score_batch(X)
    y_pred_proba = np.array([r.risk_score for r in results])
    
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(y, y_pred_proba)
    business = calculate_business_metrics(y, y_pred_proba)
    
    # Print report
    print_report(metrics, business)
    
    # Save results
    output = {
        "metrics": metrics,
        "business": business,
        "n_samples": len(X),
        "fraud_rate": float(y.mean()),
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
