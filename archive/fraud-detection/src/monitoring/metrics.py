"""
Monitoring and Metrics

Prometheus metrics, model drift detection, and alerting.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUESTS_TOTAL = Counter(
        "fraud_detection_requests_total",
        "Total number of fraud detection requests",
        ["endpoint", "status"],
    )
    
    REQUEST_DURATION = Histogram(
        "fraud_detection_request_duration_seconds",
        "Request duration in seconds",
        ["endpoint"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    
    # Model metrics
    FRAUD_SCORE = Histogram(
        "fraud_detection_score",
        "Distribution of fraud scores",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    
    DECISION_TOTAL = Counter(
        "fraud_detection_decisions_total",
        "Total decisions by type",
        ["decision"],
    )
    
    MODEL_INFERENCE_TIME = Histogram(
        "fraud_detection_model_inference_seconds",
        "Model inference time",
        ["model"],
        buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    )
    
    # Feature store metrics
    FEATURE_STORE_LATENCY = Histogram(
        "fraud_detection_feature_store_seconds",
        "Feature store latency",
        buckets=[0.0005, 0.001, 0.002, 0.005, 0.01],
    )
    
    # Drift metrics
    FEATURE_DRIFT = Gauge(
        "fraud_detection_feature_drift",
        "Feature drift score",
        ["feature"],
    )
    
    MODEL_DRIFT = Gauge(
        "fraud_detection_model_drift",
        "Model prediction drift",
    )


# =============================================================================
# Drift Detection
# =============================================================================

@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    window_size: int = 1000
    reference_size: int = 10000
    threshold: float = 0.1
    alert_cooldown_minutes: int = 60
    features_to_monitor: List[str] = field(default_factory=list)


class DriftDetector:
    """
    Detect feature and model drift.
    
    Uses Population Stability Index (PSI) for feature drift
    and prediction distribution comparison for model drift.
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        
        # Reference distributions (from training)
        self._reference_features: Dict[str, np.ndarray] = {}
        self._reference_predictions: Optional[np.ndarray] = None
        
        # Sliding windows for current data
        self._current_features: Dict[str, deque] = {}
        self._current_predictions: deque = deque(maxlen=self.config.window_size)
        
        # Alert tracking
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_callbacks: List[Callable] = []
        
    def set_reference(
        self,
        features: Dict[str, np.ndarray],
        predictions: np.ndarray,
    ):
        """Set reference distributions from training data."""
        self._reference_features = features
        self._reference_predictions = predictions
        
        # Initialize current windows
        for feature in features.keys():
            self._current_features[feature] = deque(maxlen=self.config.window_size)
            
        logger.info(f"Set reference distributions for {len(features)} features")
        
    def add_sample(
        self,
        features: Dict[str, float],
        prediction: float,
    ):
        """Add a new sample for drift monitoring."""
        # Add features
        for name, value in features.items():
            if name in self._current_features:
                self._current_features[name].append(value)
                
        # Add prediction
        self._current_predictions.append(prediction)
        
        # Check if window is full
        if len(self._current_predictions) >= self.config.window_size:
            self._check_drift()
            
    def _check_drift(self):
        """Check for drift and trigger alerts if needed."""
        now = datetime.utcnow()
        
        # Check feature drift
        for feature, current_window in self._current_features.items():
            if len(current_window) < self.config.window_size:
                continue
                
            if feature not in self._reference_features:
                continue
                
            drift_score = self._calculate_psi(
                self._reference_features[feature],
                np.array(current_window),
            )
            
            # Update Prometheus metric
            if PROMETHEUS_AVAILABLE:
                FEATURE_DRIFT.labels(feature=feature).set(drift_score)
                
            # Check threshold
            if drift_score > self.config.threshold:
                self._trigger_alert(
                    alert_type="feature_drift",
                    feature=feature,
                    score=drift_score,
                    threshold=self.config.threshold,
                )
                
        # Check model drift
        if len(self._current_predictions) >= self.config.window_size:
            if self._reference_predictions is not None:
                model_drift = self._calculate_psi(
                    self._reference_predictions,
                    np.array(self._current_predictions),
                )
                
                if PROMETHEUS_AVAILABLE:
                    MODEL_DRIFT.set(model_drift)
                    
                if model_drift > self.config.threshold:
                    self._trigger_alert(
                        alert_type="model_drift",
                        feature="predictions",
                        score=model_drift,
                        threshold=self.config.threshold,
                    )
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.25: Moderate shift
        PSI >= 0.25: Significant shift
        """
        # Create bins from reference
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Calculate distributions
        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current, bins=bins)[0]
        
        # Normalize
        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)
        
        # Avoid division by zero
        ref_pct = np.clip(ref_pct, 0.0001, 1)
        cur_pct = np.clip(cur_pct, 0.0001, 1)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    def _trigger_alert(
        self,
        alert_type: str,
        feature: str,
        score: float,
        threshold: float,
    ):
        """Trigger drift alert if not in cooldown."""
        now = datetime.utcnow()
        key = f"{alert_type}:{feature}"
        
        # Check cooldown
        if key in self._last_alert_time:
            elapsed = (now - self._last_alert_time[key]).total_seconds()
            if elapsed < self.config.alert_cooldown_minutes * 60:
                return
                
        self._last_alert_time[key] = now
        
        alert = {
            "type": alert_type,
            "feature": feature,
            "score": score,
            "threshold": threshold,
            "timestamp": now.isoformat(),
            "severity": "warning" if score < 0.25 else "critical",
        }
        
        logger.warning(f"Drift alert: {alert}")
        
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
    def register_alert_callback(self, callback: Callable):
        """Register a callback for drift alerts."""
        self._alert_callbacks.append(callback)


# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        self._latencies: deque = deque(maxlen=window_size)
        self._scores: deque = deque(maxlen=window_size)
        self._decisions: deque = deque(maxlen=window_size)
        
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        
    def record_request(
        self,
        latency_ms: float,
        score: float,
        decision: str,
        success: bool = True,
    ):
        """Record a request for monitoring."""
        self._latencies.append(latency_ms)
        self._scores.append(score)
        self._decisions.append(decision)
        
        self._request_count += 1
        if not success:
            self._error_count += 1
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        latencies = np.array(self._latencies) if self._latencies else np.array([0])
        
        elapsed = time.time() - self._start_time
        
        return {
            "requests_per_second": self._request_count / max(elapsed, 1),
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p90_latency_ms": float(np.percentile(latencies, 90)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "max_latency_ms": float(np.max(latencies)),
            "error_rate": self._error_count / max(self._request_count, 1),
            "total_requests": self._request_count,
        }
        
    def get_decision_breakdown(self) -> Dict[str, int]:
        """Get breakdown of decisions."""
        from collections import Counter
        return dict(Counter(self._decisions))


# =============================================================================
# Alert Manager
# =============================================================================

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric: str
    threshold: float
    comparison: str = "gt"  # gt, lt, eq
    severity: str = "warning"
    cooldown_minutes: int = 30


class AlertManager:
    """Manage alerts based on metrics."""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self._last_alert: Dict[str, datetime] = {}
        self._callbacks: List[Callable] = []
        
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
        
    def check_metrics(self, metrics: Dict[str, float]):
        """Check metrics against rules."""
        now = datetime.utcnow()
        
        for rule in self.rules:
            if rule.metric not in metrics:
                continue
                
            value = metrics[rule.metric]
            triggered = False
            
            if rule.comparison == "gt" and value > rule.threshold:
                triggered = True
            elif rule.comparison == "lt" and value < rule.threshold:
                triggered = True
            elif rule.comparison == "eq" and value == rule.threshold:
                triggered = True
                
            if triggered:
                # Check cooldown
                if rule.name in self._last_alert:
                    elapsed = (now - self._last_alert[rule.name]).total_seconds()
                    if elapsed < rule.cooldown_minutes * 60:
                        continue
                        
                self._last_alert[rule.name] = now
                
                alert = {
                    "rule": rule.name,
                    "metric": rule.metric,
                    "value": value,
                    "threshold": rule.threshold,
                    "severity": rule.severity,
                    "timestamp": now.isoformat(),
                }
                
                logger.warning(f"Alert triggered: {alert}")
                
                for callback in self._callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
                        
    def register_callback(self, callback: Callable):
        """Register alert callback."""
        self._callbacks.append(callback)
