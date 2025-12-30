"""
Fraud Detection Ensemble

Combines multiple models for robust fraud detection:
- XGBoost: Tabular patterns
- Neural Network: Sequential patterns
- Isolation Forest: Anomaly detection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .xgboost_model import XGBoostFraudModel, XGBoostConfig
from .neural_net import NeuralNetFraudModel, NeuralNetConfig
from .isolation_forest import IsolationForestModel, IsolationForestConfig


logger = logging.getLogger(__name__)


class Decision(Enum):
    """Transaction decision."""
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"
    STEP_UP = "step_up"  # Step-up authentication


@dataclass
class EnsembleConfig:
    """Ensemble configuration."""
    # Model weights (must sum to 1)
    xgboost_weight: float = 0.45
    neural_net_weight: float = 0.35
    isolation_forest_weight: float = 0.20
    
    # Decision thresholds
    approve_threshold: float = 0.3
    review_threshold: float = 0.7
    decline_threshold: float = 0.9
    
    # Individual model configs
    xgboost_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    neural_net_config: NeuralNetConfig = field(default_factory=NeuralNetConfig)
    isolation_forest_config: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    
    def __post_init__(self):
        # Normalize weights
        total = self.xgboost_weight + self.neural_net_weight + self.isolation_forest_weight
        self.xgboost_weight /= total
        self.neural_net_weight /= total
        self.isolation_forest_weight /= total


@dataclass
class ScoringResult:
    """Result of fraud scoring."""
    transaction_id: str
    risk_score: float
    decision: Decision
    
    # Individual model scores
    xgboost_score: float = 0.0
    neural_net_score: float = 0.0
    isolation_forest_score: float = 0.0
    
    # Explanation
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Timing
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "risk_score": self.risk_score,
            "decision": self.decision.value,
            "xgboost_score": self.xgboost_score,
            "neural_net_score": self.neural_net_score,
            "isolation_forest_score": self.isolation_forest_score,
            "top_features": self.top_features,
            "risk_factors": self.risk_factors,
            "latency_ms": self.latency_ms,
        }


class FraudEnsemble:
    """
    Ensemble model for fraud detection.
    
    Combines three models:
    1. XGBoost - Captures feature interactions
    2. Neural Network - Captures complex patterns
    3. Isolation Forest - Detects anomalies
    
    Weights are calibrated based on individual model performance.
    """
    
    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.config = config or EnsembleConfig()
        self.feature_names = feature_names
        
        # Initialize models
        self.xgboost = XGBoostFraudModel(
            config=self.config.xgboost_config,
            feature_names=feature_names,
        )
        self.neural_net = NeuralNetFraudModel(
            config=self.config.neural_net_config,
            feature_names=feature_names,
        )
        self.isolation_forest = IsolationForestModel(
            config=self.config.isolation_forest_config,
            feature_names=feature_names,
        )
        
        self._is_trained = False
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train all ensemble models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics for each model
        """
        logger.info("Training fraud detection ensemble...")
        
        metrics = {}
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        metrics["xgboost"] = self.xgboost.train(X_train, y_train, X_val, y_val)
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        metrics["neural_net"] = self.neural_net.train(X_train, y_train, X_val, y_val)
        
        # Train Isolation Forest (unsupervised)
        logger.info("Training Isolation Forest...")
        metrics["isolation_forest"] = self.isolation_forest.train(X_train)
        
        self._is_trained = True
        
        # Optionally calibrate weights on validation set
        if X_val is not None and y_val is not None:
            self._calibrate_weights(X_val, y_val)
            
        logger.info("Ensemble training complete")
        
        return metrics
    
    def _calibrate_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calibrate model weights based on validation performance."""
        from sklearn.metrics import roc_auc_score
        
        # Get predictions from each model
        xgb_preds = self.xgboost.predict_proba(X_val)
        nn_preds = self.neural_net.predict_proba(X_val)
        iso_preds = self.isolation_forest.predict_proba(X_val)
        
        # Calculate AUC for each
        xgb_auc = roc_auc_score(y_val, xgb_preds)
        nn_auc = roc_auc_score(y_val, nn_preds)
        iso_auc = roc_auc_score(y_val, iso_preds)
        
        logger.info(f"Model AUCs - XGBoost: {xgb_auc:.4f}, NN: {nn_auc:.4f}, IF: {iso_auc:.4f}")
        
        # Update weights proportionally to AUC
        total_auc = xgb_auc + nn_auc + iso_auc
        self.config.xgboost_weight = xgb_auc / total_auc
        self.config.neural_net_weight = nn_auc / total_auc
        self.config.isolation_forest_weight = iso_auc / total_auc
        
        logger.info(
            f"Calibrated weights - XGBoost: {self.config.xgboost_weight:.3f}, "
            f"NN: {self.config.neural_net_weight:.3f}, "
            f"IF: {self.config.isolation_forest_weight:.3f}"
        )
    
    def score(
        self,
        X: np.ndarray,
        transaction_id: str = "",
    ) -> ScoringResult:
        """
        Score a transaction for fraud.
        
        Args:
            X: Feature vector (1D or 2D with single row)
            transaction_id: Transaction identifier
            
        Returns:
            ScoringResult with risk score and decision
        """
        import time
        start_time = time.perf_counter()
        
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")
            
        # Ensure 2D input
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Get individual model predictions
        xgb_score = float(self.xgboost.predict_proba(X)[0])
        nn_score = float(self.neural_net.predict_proba(X)[0])
        iso_score = float(self.isolation_forest.predict_proba(X)[0])
        
        # Weighted ensemble score
        risk_score = (
            self.config.xgboost_weight * xgb_score +
            self.config.neural_net_weight * nn_score +
            self.config.isolation_forest_weight * iso_score
        )
        
        # Make decision
        decision = self._make_decision(risk_score)
        
        # Get explanation
        explanation = self.xgboost.explain(X)
        top_features = explanation.get("top_features", [])
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(X, xgb_score, nn_score, iso_score)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return ScoringResult(
            transaction_id=transaction_id,
            risk_score=risk_score,
            decision=decision,
            xgboost_score=xgb_score,
            neural_net_score=nn_score,
            isolation_forest_score=iso_score,
            top_features=top_features,
            risk_factors=risk_factors,
            latency_ms=latency_ms,
        )
    
    def score_batch(
        self,
        X: np.ndarray,
        transaction_ids: Optional[List[str]] = None,
    ) -> List[ScoringResult]:
        """
        Score multiple transactions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            transaction_ids: List of transaction IDs
            
        Returns:
            List of ScoringResults
        """
        import time
        start_time = time.perf_counter()
        
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")
            
        n_samples = X.shape[0]
        
        if transaction_ids is None:
            transaction_ids = [f"txn_{i}" for i in range(n_samples)]
            
        # Get batch predictions from each model
        xgb_scores = self.xgboost.predict_proba(X)
        nn_scores = self.neural_net.predict_proba(X)
        iso_scores = self.isolation_forest.predict_proba(X)
        
        # Calculate ensemble scores
        risk_scores = (
            self.config.xgboost_weight * xgb_scores +
            self.config.neural_net_weight * nn_scores +
            self.config.isolation_forest_weight * iso_scores
        )
        
        total_latency = (time.perf_counter() - start_time) * 1000
        per_sample_latency = total_latency / n_samples
        
        results = []
        for i in range(n_samples):
            decision = self._make_decision(risk_scores[i])
            
            results.append(ScoringResult(
                transaction_id=transaction_ids[i],
                risk_score=float(risk_scores[i]),
                decision=decision,
                xgboost_score=float(xgb_scores[i]),
                neural_net_score=float(nn_scores[i]),
                isolation_forest_score=float(iso_scores[i]),
                latency_ms=per_sample_latency,
            ))
            
        return results
    
    def _make_decision(self, risk_score: float) -> Decision:
        """Make decision based on risk score."""
        if risk_score >= self.config.decline_threshold:
            return Decision.DECLINE
        elif risk_score >= self.config.review_threshold:
            return Decision.REVIEW
        elif risk_score >= self.config.approve_threshold:
            return Decision.STEP_UP
        else:
            return Decision.APPROVE
    
    def _identify_risk_factors(
        self,
        X: np.ndarray,
        xgb_score: float,
        nn_score: float,
        iso_score: float,
    ) -> List[str]:
        """Identify human-readable risk factors."""
        factors = []
        
        # Model agreement
        if xgb_score > 0.7 and nn_score > 0.7:
            factors.append("High risk across multiple models")
            
        # Anomaly detection
        if iso_score > 0.8:
            factors.append("Transaction pattern is highly unusual")
            
        # Feature-based factors (if feature names available)
        if self.feature_names:
            feature_dict = dict(zip(self.feature_names, X.flatten()))
            
            if feature_dict.get("txn_count_1h", 0) > 10:
                factors.append("High velocity: many transactions in last hour")
                
            if feature_dict.get("is_new_merchant", 0) > 0:
                factors.append("First transaction with this merchant")
                
            if feature_dict.get("is_new_device", 0) > 0:
                factors.append("First transaction from this device")
                
            if feature_dict.get("deviation_from_avg", 0) > 3:
                factors.append("Amount significantly above average")
                
        return factors
    
    def save(self, directory: str):
        """Save all models to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        self.xgboost.save(directory / "xgboost.pkl")
        self.neural_net.save(directory / "neural_net.pt")
        self.isolation_forest.save(directory / "isolation_forest.pkl")
        
        # Save config
        import json
        with open(directory / "config.json", "w") as f:
            json.dump({
                "xgboost_weight": self.config.xgboost_weight,
                "neural_net_weight": self.config.neural_net_weight,
                "isolation_forest_weight": self.config.isolation_forest_weight,
                "approve_threshold": self.config.approve_threshold,
                "review_threshold": self.config.review_threshold,
                "decline_threshold": self.config.decline_threshold,
            }, f)
            
        logger.info(f"Ensemble saved to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> "FraudEnsemble":
        """Load ensemble from directory."""
        directory = Path(directory)
        
        # Load config
        import json
        with open(directory / "config.json") as f:
            config_dict = json.load(f)
            
        config = EnsembleConfig(
            xgboost_weight=config_dict["xgboost_weight"],
            neural_net_weight=config_dict["neural_net_weight"],
            isolation_forest_weight=config_dict["isolation_forest_weight"],
            approve_threshold=config_dict["approve_threshold"],
            review_threshold=config_dict["review_threshold"],
            decline_threshold=config_dict["decline_threshold"],
        )
        
        instance = cls(config=config)
        
        # Load individual models
        instance.xgboost = XGBoostFraudModel.load(directory / "xgboost.pkl")
        instance.neural_net = NeuralNetFraudModel.load(directory / "neural_net.pt")
        instance.isolation_forest = IsolationForestModel.load(directory / "isolation_forest.pkl")
        
        instance._is_trained = True
        
        logger.info(f"Ensemble loaded from {directory}")
        return instance
