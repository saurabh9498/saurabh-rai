"""
Evaluation metrics for recommendation systems.

Includes ranking metrics (NDCG, MRR, MAP), classification metrics (AUC, Log Loss),
and business metrics (CTR, Revenue).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    
    # Ranking metrics
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Classification metrics
    auc: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    
    # Business metrics
    ctr: float = 0.0
    conversion_rate: float = 0.0
    revenue_per_session: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ranking': {
                'ndcg': self.ndcg_at_k,
                'mrr': self.mrr,
                'map': self.map_score,
                'hit_rate': self.hit_rate_at_k,
            },
            'classification': {
                'auc': self.auc,
                'log_loss': self.log_loss,
                'accuracy': self.accuracy,
            },
            'business': {
                'ctr': self.ctr,
                'conversion_rate': self.conversion_rate,
                'revenue_per_session': self.revenue_per_session,
            },
        }


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at K."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.
    
    Args:
        y_true: Ground truth relevance scores
        y_pred: Predicted scores for ranking
        k: Number of top items to consider
        
    Returns:
        NDCG@K score between 0 and 1
    """
    # Sort by predicted scores
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order)
    
    # Compute DCG
    dcg = dcg_at_k(y_true_sorted, k)
    
    # Compute ideal DCG (sort by true relevance)
    ideal_order = np.argsort(y_true)[::-1]
    ideal_relevances = np.take(y_true, ideal_order)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        y_true: Binary relevance labels (1 = relevant)
        y_pred: Predicted scores for ranking
        
    Returns:
        MRR score
    """
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order)
    
    # Find first relevant item
    relevant_indices = np.where(y_true_sorted > 0)[0]
    
    if len(relevant_indices) == 0:
        return 0.0
    
    first_relevant = relevant_indices[0]
    return 1.0 / (first_relevant + 1)


def hit_rate_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Hit Rate at K (fraction of users with at least one hit in top-K).
    
    Args:
        y_true: Binary relevance labels
        y_pred: Predicted scores for ranking
        k: Number of top items to consider
        
    Returns:
        Hit rate between 0 and 1
    """
    order = np.argsort(y_pred)[::-1][:k]
    hits = np.take(y_true, order)
    return float(np.any(hits > 0))


def precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> float:
    """Compute Precision at K."""
    order = np.argsort(y_pred)[::-1][:k]
    relevant = np.take(y_true, order)
    return np.sum(relevant > 0) / k


def recall_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> float:
    """Compute Recall at K."""
    total_relevant = np.sum(y_true > 0)
    if total_relevant == 0:
        return 0.0
    
    order = np.argsort(y_pred)[::-1][:k]
    relevant = np.take(y_true, order)
    return np.sum(relevant > 0) / total_relevant


def average_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Average Precision for a single query."""
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order)
    
    relevant_mask = y_true_sorted > 0
    if not np.any(relevant_mask):
        return 0.0
    
    precision_at_relevant = []
    relevant_count = 0
    
    for i, is_relevant in enumerate(relevant_mask):
        if is_relevant:
            relevant_count += 1
            precision_at_relevant.append(relevant_count / (i + 1))
    
    return np.mean(precision_at_relevant)


def auc_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Area Under ROC Curve."""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)
    except ImportError:
        logger.warning("sklearn not available, using manual AUC computation")
        return _manual_auc(y_true, y_pred)


def _manual_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Manual AUC computation without sklearn."""
    # Sort by predicted score
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    
    # Count positive and negative examples
    n_pos = np.sum(y_true_sorted)
    n_neg = len(y_true_sorted) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Compute AUC using ranking
    rank_sum = np.sum(np.where(y_true_sorted)[0])
    auc = (rank_sum - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    
    return 1 - auc  # Flip because we sorted descending


def log_loss_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute Log Loss (Binary Cross-Entropy)."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: List[int] = [5, 10, 20, 50],
) -> MetricsResult:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted scores
        k_values: K values for ranking metrics
        
    Returns:
        MetricsResult containing all metrics
    """
    result = MetricsResult()
    
    # Ranking metrics
    for k in k_values:
        result.ndcg_at_k[k] = ndcg_at_k(y_true, y_pred, k)
        result.hit_rate_at_k[k] = hit_rate_at_k(y_true, y_pred, k)
    
    result.mrr = mrr_score(y_true, y_pred)
    result.map_score = average_precision(y_true, y_pred)
    
    # Classification metrics
    result.auc = auc_score(y_true, y_pred)
    result.log_loss = log_loss_score(y_true, y_pred)
    result.accuracy = np.mean((y_pred >= 0.5) == y_true)
    
    return result


class OnlineMetricsTracker:
    """Track online recommendation metrics."""
    
    def __init__(self):
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.revenue = 0.0
        self.latencies = []
    
    def record_impression(self):
        """Record an impression."""
        self.impressions += 1
    
    def record_click(self):
        """Record a click."""
        self.clicks += 1
    
    def record_conversion(self, revenue: float = 0.0):
        """Record a conversion with optional revenue."""
        self.conversions += 1
        self.revenue += revenue
    
    def record_latency(self, latency_ms: float):
        """Record request latency."""
        self.latencies.append(latency_ms)
    
    @property
    def ctr(self) -> float:
        """Click-through rate."""
        return self.clicks / max(self.impressions, 1)
    
    @property
    def conversion_rate(self) -> float:
        """Conversion rate."""
        return self.conversions / max(self.clicks, 1)
    
    @property
    def revenue_per_impression(self) -> float:
        """Revenue per impression."""
        return self.revenue / max(self.impressions, 1)
    
    @property
    def p50_latency(self) -> float:
        """P50 latency in ms."""
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 50))
    
    @property
    def p99_latency(self) -> float:
        """P99 latency in ms."""
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 99))
    
    def get_summary(self) -> Dict[str, float]:
        """Get metrics summary."""
        return {
            'impressions': self.impressions,
            'clicks': self.clicks,
            'conversions': self.conversions,
            'revenue': self.revenue,
            'ctr': self.ctr,
            'conversion_rate': self.conversion_rate,
            'revenue_per_impression': self.revenue_per_impression,
            'p50_latency_ms': self.p50_latency,
            'p99_latency_ms': self.p99_latency,
        }
    
    def reset(self):
        """Reset all metrics."""
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.revenue = 0.0
        self.latencies = []
