"""
Isolation Forest Anomaly Detector

Unsupervised anomaly detection for:
- Detecting novel fraud patterns
- Zero-day fraud detection
- Complementing supervised models
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class IsolationForestConfig:
    """Isolation Forest configuration."""
    n_estimators: int = 200
    contamination: float = 0.01  # Expected fraud rate
    max_samples: str = "auto"
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = -1
    random_state: int = 42
    warm_start: bool = False


class IsolationForestModel:
    """
    Isolation Forest for anomaly-based fraud detection.
    
    Strengths:
    - Unsupervised (no labels needed)
    - Detects novel fraud patterns
    - Fast training and inference
    - Good for zero-day fraud
    
    How it works:
    - Isolates anomalies by random partitioning
    - Anomalies require fewer splits to isolate
    - Anomaly score based on path length
    """
    
    def __init__(
        self,
        config: Optional[IsolationForestConfig] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.config = config or IsolationForestConfig()
        self.feature_names = feature_names
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,  # Ignored (unsupervised)
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the Isolation Forest.
        
        Note: This is unsupervised, so labels are ignored.
        We train on all data (including fraud) to learn what's normal.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
            
        logger.info("Training Isolation Forest anomaly detector...")
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            warm_start=self.config.warm_start,
        )
        
        self.model.fit(X_scaled)
        
        # Calculate baseline statistics
        scores = self._get_raw_scores(X_scaled)
        
        metrics = {
            "n_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "anomaly_threshold": float(np.percentile(scores, 99)),
        }
        
        logger.info(f"Training complete. Mean anomaly score: {metrics['mean_score']:.4f}")
        
        return metrics
    
    def _get_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Get raw anomaly scores (higher = more anomalous)."""
        # decision_function returns negative for anomalies
        # We negate so higher = more anomalous
        return -self.model.decision_function(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly probability (0-1).
        
        Converts raw anomaly scores to probabilities using
        sigmoid transformation.
        
        Args:
            X: Features array
            
        Returns:
            Anomaly probabilities (0-1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
            
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Get raw scores
        raw_scores = self._get_raw_scores(X_scaled)
        
        # Convert to probability using sigmoid
        # Shift and scale to center around 0.5 for typical transactions
        probabilities = 1 / (1 + np.exp(-5 * (raw_scores - 0.5)))
        
        return np.clip(probabilities, 0, 1)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions (1 = anomaly/fraud)."""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw anomaly scores.
        
        Returns:
            Raw scores (higher = more anomalous)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
            
        X_scaled = self.scaler.transform(X)
        return self._get_raw_scores(X_scaled)
    
    def partial_fit(self, X: np.ndarray):
        """
        Incrementally update the model with new data.
        
        Uses warm_start to add new trees.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
            
        if not self.config.warm_start:
            logger.warning("warm_start not enabled, skipping partial fit")
            return
            
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)
        
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "config": self.config,
                "feature_names": self.feature_names,
            }, f)
            
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "IsolationForestModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        instance = cls(
            config=data["config"],
            feature_names=data["feature_names"],
        )
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        
        logger.info(f"Model loaded from {path}")
        return instance


class MockIsolationForestModel(IsolationForestModel):
    """Mock Isolation Forest for testing."""
    
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        return {"n_samples": 1000, "mean_score": 0.1}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        # Most samples are normal, few are anomalies
        return np.random.beta(0.5, 10, n_samples)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        return np.random.uniform(-0.5, 0.5, n_samples)
