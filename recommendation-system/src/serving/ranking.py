"""
DLRM Ranking Service

GPU-accelerated ranking using DLRM model via Triton Inference Server.
Handles batch scoring, multi-objective optimization, and score calibration.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class RankingRequest:
    """Request for ranking candidates."""
    user_id: str
    candidate_items: List[str]
    user_features: Dict[str, Any]
    item_features: Dict[str, Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None


@dataclass
class RankedItem:
    """Ranked item with scores."""
    item_id: str
    score: float
    ctr_score: float = 0.0
    cvr_score: float = 0.0
    revenue_score: float = 0.0
    rank: int = 0
    reason: str = ""


@dataclass
class RankingResponse:
    """Response from ranking service."""
    items: List[RankedItem]
    latency_ms: float = 0.0
    model_version: str = ""


class FeatureAssembler:
    """Assembles features for DLRM model."""
    
    def __init__(
        self,
        sparse_feature_names: List[str],
        dense_feature_names: List[str],
    ):
        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names = dense_feature_names
    
    def assemble(
        self,
        user_features: Dict[str, Any],
        item_features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble features into model input format.
        
        Returns:
            sparse_features: (num_sparse,) int array
            dense_features: (num_dense,) float array
        """
        context = context or {}
        
        # Combine all feature sources
        all_features = {**user_features, **item_features, **context}
        
        # Extract sparse features
        sparse = []
        for name in self.sparse_feature_names:
            value = all_features.get(name, 0)
            sparse.append(int(value))
        
        # Extract dense features
        dense = []
        for name in self.dense_feature_names:
            value = all_features.get(name, 0.0)
            dense.append(float(value))
        
        return np.array(sparse, dtype=np.int64), np.array(dense, dtype=np.float32)
    
    def batch_assemble(
        self,
        user_features: Dict[str, Any],
        item_features_list: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble features for a batch of items."""
        sparse_batch = []
        dense_batch = []
        
        for item_features in item_features_list:
            sparse, dense = self.assemble(user_features, item_features, context)
            sparse_batch.append(sparse)
            dense_batch.append(dense)
        
        return (
            np.stack(sparse_batch),
            np.stack(dense_batch),
        )


class DLRMRankingService:
    """Ranking service using DLRM model."""
    
    def __init__(
        self,
        triton_client: Any,  # TritonClient instance
        model_name: str = "dlrm",
        feature_assembler: Optional[FeatureAssembler] = None,
        batch_size: int = 64,
    ):
        self.triton_client = triton_client
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.feature_assembler = feature_assembler or FeatureAssembler(
            sparse_feature_names=[
                'user_id', 'item_id', 'category_l2', 'brand_id',
                'age_bucket', 'gender', 'country', 'device_type',
                'page_type', 'hour_of_day',
            ],
            dense_feature_names=[
                'price', 'rating', 'review_count', 'discount_pct',
                'ctr_7d', 'conversion_rate_7d', 'account_age_days',
                'total_orders', 'days_since_last_order',
                'session_duration_sec', 'items_in_cart', 'cart_value',
            ],
        )
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def rank(
        self,
        request: RankingRequest,
    ) -> RankingResponse:
        """
        Rank candidate items for a user.
        
        Args:
            request: Ranking request with candidates and features
            
        Returns:
            RankingResponse with ranked items
        """
        import time
        start_time = time.time()
        
        # Prepare features for all candidates
        item_features_list = [
            request.item_features.get(item_id, {})
            for item_id in request.candidate_items
        ]
        
        sparse_features, dense_features = self.feature_assembler.batch_assemble(
            request.user_features,
            item_features_list,
            request.context,
        )
        
        # Score in batches
        all_scores = []
        
        for i in range(0, len(request.candidate_items), self.batch_size):
            batch_sparse = sparse_features[i:i + self.batch_size]
            batch_dense = dense_features[i:i + self.batch_size]
            
            # Call Triton
            scores = await self._score_batch(batch_sparse, batch_dense)
            all_scores.extend(scores)
        
        # Create ranked items
        scored_items = list(zip(request.candidate_items, all_scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        ranked_items = [
            RankedItem(
                item_id=item_id,
                score=float(score),
                ctr_score=float(score),  # Could be multi-task
                rank=rank + 1,
            )
            for rank, (item_id, score) in enumerate(scored_items)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RankingResponse(
            items=ranked_items,
            latency_ms=latency_ms,
            model_version=self.model_name,
        )
    
    async def _score_batch(
        self,
        sparse_features: np.ndarray,
        dense_features: np.ndarray,
    ) -> List[float]:
        """Score a batch using Triton."""
        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                self.executor,
                self._triton_infer,
                sparse_features,
                dense_features,
            )
            return scores.tolist()
        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            # Return neutral scores on failure
            return [0.5] * len(sparse_features)
    
    def _triton_infer(
        self,
        sparse_features: np.ndarray,
        dense_features: np.ndarray,
    ) -> np.ndarray:
        """Synchronous Triton inference."""
        if self.triton_client is None:
            # Mock inference for testing
            return np.random.rand(len(sparse_features))
        
        # Actual Triton inference
        return self.triton_client.infer(
            model_name=self.model_name,
            inputs={
                'sparse_features': sparse_features,
                'dense_features': dense_features,
            },
        )['output']


class MultiObjectiveRanker:
    """
    Multi-objective ranking with configurable weights.
    
    Combines CTR, CVR, revenue, and diversity objectives.
    """
    
    def __init__(
        self,
        ctr_weight: float = 0.5,
        cvr_weight: float = 0.3,
        revenue_weight: float = 0.2,
        diversity_weight: float = 0.0,
    ):
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight
        self.revenue_weight = revenue_weight
        self.diversity_weight = diversity_weight
    
    def compute_final_scores(
        self,
        items: List[RankedItem],
        item_categories: Optional[Dict[str, int]] = None,
    ) -> List[RankedItem]:
        """Compute final scores with multi-objective optimization."""
        
        # Combine objective scores
        for item in items:
            base_score = (
                self.ctr_weight * item.ctr_score +
                self.cvr_weight * item.cvr_score +
                self.revenue_weight * item.revenue_score
            )
            item.score = base_score
        
        # Apply diversity re-ranking if enabled
        if self.diversity_weight > 0 and item_categories:
            items = self._diversify(items, item_categories)
        
        # Re-rank by final score
        items.sort(key=lambda x: x.score, reverse=True)
        for i, item in enumerate(items):
            item.rank = i + 1
        
        return items
    
    def _diversify(
        self,
        items: List[RankedItem],
        item_categories: Dict[str, int],
    ) -> List[RankedItem]:
        """Apply MMR-style diversity re-ranking."""
        if not items:
            return items
        
        selected = [items[0]]
        remaining = items[1:]
        
        while remaining and len(selected) < len(items):
            best_item = None
            best_score = float('-inf')
            
            for item in remaining:
                # Compute diversity penalty
                item_cat = item_categories.get(item.item_id)
                
                max_similarity = 0.0
                for sel_item in selected:
                    sel_cat = item_categories.get(sel_item.item_id)
                    if item_cat == sel_cat:
                        max_similarity = 1.0
                        break
                
                # MMR score
                mmr_score = (
                    (1 - self.diversity_weight) * item.score -
                    self.diversity_weight * max_similarity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = item
            
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
        
        return selected


class ScoreCalibrator:
    """Calibrates model scores to probabilities."""
    
    def __init__(self, method: str = "platt"):
        self.method = method
        self.a = 1.0
        self.b = 0.0
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit calibration parameters."""
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            a, b = params
            calibrated = 1 / (1 + np.exp(-(a * scores + b)))
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
            return -np.sum(
                labels * np.log(calibrated) +
                (1 - labels) * np.log(1 - calibrated)
            )
        
        result = minimize(neg_log_likelihood, [1.0, 0.0], method='L-BFGS-B')
        self.a, self.b = result.x
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        return 1 / (1 + np.exp(-(self.a * scores + self.b)))
