"""
XGBoost Fraud Classifier

Gradient boosting model optimized for fraud detection with:
- Class imbalance handling
- Feature importance tracking
- SHAP explanations
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class XGBoostConfig:
    """XGBoost model configuration."""
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    scale_pos_weight: float = 10.0  # Handle class imbalance
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    early_stopping_rounds: int = 50
    random_state: int = 42


class XGBoostFraudModel:
    """
    XGBoost classifier for fraud detection.
    
    Strengths:
    - Excellent for tabular data with feature interactions
    - Handles missing values naturally
    - Fast inference
    - Built-in feature importance
    """
    
    def __init__(
        self,
        config: Optional[XGBoostConfig] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.config = config or XGBoostConfig()
        self.feature_names = feature_names
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[Any] = None
        self._feature_importance: Optional[Dict[str, float]] = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost not available")
            
        logger.info("Training XGBoost fraud classifier...")
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            scale_pos_weight=self.config.scale_pos_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective=self.config.objective,
            eval_metric=self.config.eval_metric,
            random_state=self.config.random_state,
            use_label_encoder=False,
            tree_method="hist",  # Fast histogram-based
            device="cpu",
        )
        
        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=100,
        )
        
        # Cache feature importance
        self._compute_feature_importance()
        
        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)
            
        metrics = {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "n_features": X_train.shape[1],
        }
        
        logger.info(f"Training complete. Best iteration: {metrics['best_iteration']}")
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get fraud probability.
        
        Args:
            X: Features array
            
        Returns:
            Fraud probabilities (0-1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
            
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary predictions.
        
        Args:
            X: Features array
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get SHAP explanations for predictions.
        
        Args:
            X: Features array
            
        Returns:
            SHAP values and base values
        """
        if self.explainer is None:
            return {}
            
        shap_values = self.explainer.shap_values(X)
        
        explanations = {
            "shap_values": shap_values,
            "base_value": self.explainer.expected_value,
            "feature_names": self.feature_names,
        }
        
        # Top contributing features
        if self.feature_names and len(X.shape) == 1:
            importance = list(zip(self.feature_names, shap_values))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            explanations["top_features"] = importance[:5]
            
        return explanations
    
    def _compute_feature_importance(self):
        """Compute and cache feature importance."""
        if self.model is None:
            return
            
        importance = self.model.feature_importances_
        
        if self.feature_names:
            self._feature_importance = dict(zip(self.feature_names, importance))
        else:
            self._feature_importance = {f"f{i}": v for i, v in enumerate(importance)}
            
        # Sort by importance
        self._feature_importance = dict(
            sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
    
    @property
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance dictionary."""
        if self._feature_importance is None:
            self._compute_feature_importance()
        return self._feature_importance or {}
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "config": self.config,
                "feature_names": self.feature_names,
                "feature_importance": self._feature_importance,
            }, f)
            
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "XGBoostFraudModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        instance = cls(
            config=data["config"],
            feature_names=data["feature_names"],
        )
        instance.model = data["model"]
        instance._feature_importance = data["feature_importance"]
        
        # Reinitialize SHAP explainer
        if SHAP_AVAILABLE and instance.model:
            instance.explainer = shap.TreeExplainer(instance.model)
            
        logger.info(f"Model loaded from {path}")
        return instance


class MockXGBoostModel(XGBoostFraudModel):
    """Mock XGBoost model for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        return {"best_iteration": 100, "best_score": 0.95}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Return realistic-looking probabilities
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        return np.random.beta(1, 20, n_samples)  # Skewed toward low values
    
    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        return {"top_features": [("amount", 0.3), ("velocity", 0.2)]}
