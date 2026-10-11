"""
A/B Testing Framework for Recommendations

Provides experiment management, variant assignment, and metrics tracking.

Usage:
    from src.serving.ab_testing import ExperimentClient
    
    experiment = ExperimentClient("homepage_recs_v2")
    variant = experiment.get_variant(user_id)
    
    if variant == "treatment":
        # Use new model
        pass
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from enum import Enum

logger = logging.getLogger(__name__)


class VariantType(Enum):
    """Experiment variant types."""
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class Variant:
    """Experiment variant configuration."""
    name: str
    weight: float  # 0.0 to 1.0
    model: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """A/B test experiment configuration."""
    name: str
    description: str = ""
    enabled: bool = True
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    variants: List[Variant] = field(default_factory=list)
    
    # Targeting
    user_segments: List[str] = field(default_factory=list)
    min_user_percentage: float = 0.0
    max_user_percentage: float = 100.0
    
    # Metrics
    primary_metric: str = "ctr"
    secondary_metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.variants:
            self.variants = [
                Variant(name="control", weight=0.5),
                Variant(name="treatment", weight=0.5),
            ]
    
    @property
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        if not self.enabled:
            return False
        
        now = datetime.now()
        
        if self.start_time and now < self.start_time:
            return False
        
        if self.end_time and now > self.end_time:
            return False
        
        return True


class ExperimentClient:
    """Client for A/B test experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        salt: str = "recommendation_ab_test",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_name = experiment_name
        self.salt = salt
        self.experiment = self._load_experiment(experiment_name, config)
    
    def _load_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Load experiment configuration."""
        if config:
            return Experiment(
                name=name,
                **config.get(name, {}),
            )
        
        # Default experiment
        return Experiment(
            name=name,
            variants=[
                Variant(name="control", weight=0.5, model="dlrm_v1"),
                Variant(name="treatment", weight=0.5, model="dlrm_v2"),
            ],
        )
    
    def get_bucket(self, user_id: str) -> float:
        """
        Get deterministic bucket for user (0.0 to 1.0).
        
        Uses consistent hashing to ensure same user always gets same bucket.
        """
        hash_input = f"{self.salt}:{self.experiment_name}:{user_id}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], 'big')
        return hash_int / (2**64)
    
    def get_variant(self, user_id: str) -> str:
        """
        Get variant assignment for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Variant name (e.g., "control", "treatment")
        """
        if not self.experiment.is_active:
            return "control"
        
        bucket = self.get_bucket(user_id)
        
        cumulative_weight = 0.0
        for variant in self.experiment.variants:
            cumulative_weight += variant.weight
            if bucket < cumulative_weight:
                return variant.name
        
        # Fallback to last variant
        return self.experiment.variants[-1].name
    
    def get_variant_config(self, user_id: str) -> Variant:
        """Get full variant configuration for user."""
        variant_name = self.get_variant(user_id)
        
        for variant in self.experiment.variants:
            if variant.name == variant_name:
                return variant
        
        return self.experiment.variants[0]
    
    def get_model(self, user_id: str) -> Optional[str]:
        """Get model to use for user based on variant."""
        variant = self.get_variant_config(user_id)
        return variant.model
    
    def is_treatment(self, user_id: str) -> bool:
        """Check if user is in treatment group."""
        return self.get_variant(user_id) != "control"
    
    def log_exposure(
        self,
        user_id: str,
        request_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Log experiment exposure for analysis."""
        variant = self.get_variant(user_id)
        
        exposure = {
            "experiment": self.experiment_name,
            "variant": variant,
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        }
        
        logger.info(f"Experiment exposure: {json.dumps(exposure)}")
        
        # In production, would send to analytics system
        # analytics.track("experiment_exposure", exposure)


class ExperimentManager:
    """Manages multiple A/B test experiments."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.experiments: Dict[str, ExperimentClient] = {}
    
    def get_experiment(self, name: str) -> ExperimentClient:
        """Get or create experiment client."""
        if name not in self.experiments:
            self.experiments[name] = ExperimentClient(name, config=self.config)
        return self.experiments[name]
    
    def get_all_variants(self, user_id: str) -> Dict[str, str]:
        """Get variant assignments for all experiments."""
        return {
            name: client.get_variant(user_id)
            for name, client in self.experiments.items()
        }
    
    def register_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        **kwargs,
    ):
        """Register a new experiment."""
        config = {
            name: {
                "variants": [Variant(**v) for v in variants],
                **kwargs,
            }
        }
        self.experiments[name] = ExperimentClient(name, config=config)


class MetricsCollector:
    """Collects metrics for A/B test analysis."""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, List[float]]] = {}
    
    def record(
        self,
        experiment: str,
        variant: str,
        metric_name: str,
        value: float,
    ):
        """Record a metric value."""
        key = f"{experiment}:{variant}"
        
        if key not in self.metrics:
            self.metrics[key] = {}
        
        if metric_name not in self.metrics[key]:
            self.metrics[key][metric_name] = []
        
        self.metrics[key][metric_name].append(value)
    
    def get_summary(
        self,
        experiment: str,
        metric_name: str,
    ) -> Dict[str, Dict[str, float]]:
        """Get metric summary by variant."""
        import numpy as np
        
        summary = {}
        
        for key, metrics in self.metrics.items():
            if not key.startswith(f"{experiment}:"):
                continue
            
            variant = key.split(":")[1]
            
            if metric_name in metrics:
                values = np.array(metrics[metric_name])
                summary[variant] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        
        return summary
    
    def compute_significance(
        self,
        experiment: str,
        metric_name: str,
        control_variant: str = "control",
    ) -> Dict[str, float]:
        """Compute statistical significance between variants."""
        from scipy import stats
        import numpy as np
        
        control_key = f"{experiment}:{control_variant}"
        
        if control_key not in self.metrics:
            return {}
        
        if metric_name not in self.metrics[control_key]:
            return {}
        
        control_values = np.array(self.metrics[control_key][metric_name])
        
        results = {}
        
        for key, metrics in self.metrics.items():
            if not key.startswith(f"{experiment}:"):
                continue
            
            variant = key.split(":")[1]
            
            if variant == control_variant:
                continue
            
            if metric_name not in metrics:
                continue
            
            treatment_values = np.array(metrics[metric_name])
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((control_values.std()**2 + treatment_values.std()**2) / 2)
            cohens_d = (treatment_values.mean() - control_values.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Relative lift
            lift = (treatment_values.mean() - control_values.mean()) / control_values.mean() if control_values.mean() > 0 else 0
            
            results[variant] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "cohens_d": float(cohens_d),
                "relative_lift": float(lift),
            }
        
        return results


# Global experiment manager instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get global experiment manager."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager


def get_variant(experiment_name: str, user_id: str) -> str:
    """Convenience function to get variant for user."""
    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_name)
    return experiment.get_variant(user_id)
