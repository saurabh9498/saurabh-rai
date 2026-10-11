"""
Item Feature Computation

Computes and manages item-level features for recommendations:
- Static catalog attributes
- Dynamic popularity metrics
- Text and image embeddings
- Freshness and inventory signals
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ItemProfile:
    """Item profile with computed features."""
    item_id: str
    
    # Catalog attributes
    category_l1: int = 0
    category_l2: int = 0
    category_l3: int = 0
    brand_id: int = 0
    seller_id: int = 0
    
    # Pricing
    price: float = 0.0
    original_price: float = 0.0
    discount_pct: float = 0.0
    price_bucket: int = 0
    
    # Quality signals
    rating: float = 0.0
    review_count: int = 0
    rating_count: int = 0
    
    # Popularity metrics
    view_count_7d: int = 0
    purchase_count_7d: int = 0
    ctr_7d: float = 0.0
    conversion_rate_7d: float = 0.0
    
    # Freshness
    days_since_listed: int = 0
    is_new: bool = False
    
    # Inventory
    in_stock: bool = True
    inventory_level: int = 0
    
    # Embeddings (precomputed)
    title_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'category_l1': self.category_l1,
            'category_l2': self.category_l2,
            'category_l3': self.category_l3,
            'brand_id': self.brand_id,
            'price': self.price,
            'discount_pct': self.discount_pct,
            'rating': self.rating,
            'review_count': self.review_count,
            'view_count_7d': self.view_count_7d,
            'purchase_count_7d': self.purchase_count_7d,
            'ctr_7d': self.ctr_7d,
            'days_since_listed': self.days_since_listed,
            'in_stock': self.in_stock,
        }
    
    def to_dense_features(self) -> List[float]:
        """Convert to dense feature vector."""
        return [
            self.price,
            self.original_price,
            self.discount_pct,
            self.rating,
            np.log1p(self.review_count),
            np.log1p(self.view_count_7d),
            np.log1p(self.purchase_count_7d),
            self.ctr_7d,
            self.conversion_rate_7d,
            np.log1p(self.days_since_listed),
            float(self.in_stock),
            np.log1p(self.inventory_level),
        ]


class ItemFeatureComputer:
    """Computes item features from raw data."""
    
    def __init__(
        self,
        price_buckets: List[float] = None,
        new_item_days: int = 7,
    ):
        self.price_buckets = price_buckets or [
            10, 25, 50, 100, 200, 500, 1000, float('inf')
        ]
        self.new_item_days = new_item_days
    
    def compute_features(
        self,
        item_id: str,
        catalog_data: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
    ) -> ItemProfile:
        """
        Compute all item features.
        
        Args:
            item_id: Item identifier
            catalog_data: Static catalog attributes
            stats: Aggregated statistics (views, purchases, etc.)
            
        Returns:
            ItemProfile with computed features
        """
        profile = ItemProfile(item_id=item_id)
        
        # Catalog attributes
        profile.category_l1 = catalog_data.get('category_l1_id', 0)
        profile.category_l2 = catalog_data.get('category_l2_id', 0)
        profile.category_l3 = catalog_data.get('category_l3_id', 0)
        profile.brand_id = catalog_data.get('brand_id', 0)
        profile.seller_id = catalog_data.get('seller_id', 0)
        
        # Pricing
        profile.price = catalog_data.get('price', 0.0)
        profile.original_price = catalog_data.get('original_price', profile.price)
        profile.discount_pct = self._compute_discount(
            profile.price, profile.original_price
        )
        profile.price_bucket = self._compute_price_bucket(profile.price)
        
        # Quality
        profile.rating = catalog_data.get('rating', 0.0)
        profile.review_count = catalog_data.get('review_count', 0)
        profile.rating_count = catalog_data.get('rating_count', 0)
        
        # Freshness
        listed_date = catalog_data.get('listed_at')
        profile.days_since_listed = self._compute_days_since(listed_date)
        profile.is_new = profile.days_since_listed <= self.new_item_days
        
        # Inventory
        profile.in_stock = catalog_data.get('in_stock', True)
        profile.inventory_level = catalog_data.get('inventory_level', 0)
        
        # Popularity (from stats)
        if stats:
            profile.view_count_7d = stats.get('views_7d', 0)
            profile.purchase_count_7d = stats.get('purchases_7d', 0)
            profile.ctr_7d = self._compute_ctr(stats)
            profile.conversion_rate_7d = self._compute_conversion_rate(stats)
        
        return profile
    
    def _compute_discount(self, price: float, original_price: float) -> float:
        """Compute discount percentage."""
        if original_price <= 0 or price >= original_price:
            return 0.0
        return (original_price - price) / original_price
    
    def _compute_price_bucket(self, price: float) -> int:
        """Assign price to bucket."""
        for i, boundary in enumerate(self.price_buckets):
            if price < boundary:
                return i
        return len(self.price_buckets)
    
    def _compute_days_since(self, date_str: Optional[str]) -> int:
        """Compute days since a date."""
        if not date_str:
            return 0
        
        try:
            date = datetime.fromisoformat(date_str)
            return (datetime.now() - date).days
        except:
            return 0
    
    def _compute_ctr(self, stats: Dict[str, Any]) -> float:
        """Compute click-through rate."""
        impressions = stats.get('impressions_7d', 0)
        clicks = stats.get('clicks_7d', 0)
        return clicks / max(impressions, 1)
    
    def _compute_conversion_rate(self, stats: Dict[str, Any]) -> float:
        """Compute conversion rate."""
        clicks = stats.get('clicks_7d', 0)
        purchases = stats.get('purchases_7d', 0)
        return purchases / max(clicks, 1)


class ItemPopularityTracker:
    """Tracks real-time item popularity."""
    
    def __init__(
        self,
        decay_factor: float = 0.95,
        update_interval_sec: int = 60,
    ):
        self.decay_factor = decay_factor
        self.update_interval_sec = update_interval_sec
        
        self.view_counts = {}
        self.click_counts = {}
        self.purchase_counts = {}
        self.last_update = datetime.now()
    
    def record_event(self, item_id: str, event_type: str):
        """Record an event for an item."""
        self._maybe_decay()
        
        if event_type == 'view':
            self.view_counts[item_id] = self.view_counts.get(item_id, 0) + 1
        elif event_type == 'click':
            self.click_counts[item_id] = self.click_counts.get(item_id, 0) + 1
        elif event_type == 'purchase':
            self.purchase_counts[item_id] = self.purchase_counts.get(item_id, 0) + 1
    
    def _maybe_decay(self):
        """Apply decay to counts periodically."""
        now = datetime.now()
        elapsed = (now - self.last_update).seconds
        
        if elapsed >= self.update_interval_sec:
            for item_id in self.view_counts:
                self.view_counts[item_id] *= self.decay_factor
            for item_id in self.click_counts:
                self.click_counts[item_id] *= self.decay_factor
            for item_id in self.purchase_counts:
                self.purchase_counts[item_id] *= self.decay_factor
            
            self.last_update = now
    
    def get_popularity_score(self, item_id: str) -> float:
        """Get popularity score for an item."""
        views = self.view_counts.get(item_id, 0)
        clicks = self.click_counts.get(item_id, 0)
        purchases = self.purchase_counts.get(item_id, 0)
        
        # Weighted combination
        return views * 1 + clicks * 5 + purchases * 20
    
    def get_top_items(self, k: int = 100) -> List[Tuple[str, float]]:
        """Get top-k most popular items."""
        all_items = set(self.view_counts.keys()) | set(self.click_counts.keys())
        
        scored_items = [
            (item_id, self.get_popularity_score(item_id))
            for item_id in all_items
        ]
        
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:k]


class ItemSimilarityComputer:
    """Computes item similarities for content-based recommendations."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.item_embeddings = {}
    
    def add_embedding(self, item_id: str, embedding: np.ndarray):
        """Add or update item embedding."""
        self.item_embeddings[item_id] = embedding / (
            np.linalg.norm(embedding) + 1e-8
        )
    
    def get_similar_items(
        self,
        item_id: str,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find k most similar items."""
        if item_id not in self.item_embeddings:
            return []
        
        query_embedding = self.item_embeddings[item_id]
        exclude_set = set(exclude_items or []) | {item_id}
        
        similarities = []
        for other_id, other_embedding in self.item_embeddings.items():
            if other_id not in exclude_set:
                sim = np.dot(query_embedding, other_embedding)
                similarities.append((other_id, float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def batch_compute_similarities(
        self,
        item_ids: List[str],
    ) -> np.ndarray:
        """Compute pairwise similarities for a batch of items."""
        embeddings = []
        valid_ids = []
        
        for item_id in item_ids:
            if item_id in self.item_embeddings:
                embeddings.append(self.item_embeddings[item_id])
                valid_ids.append(item_id)
        
        if not embeddings:
            return np.array([])
        
        embedding_matrix = np.stack(embeddings)
        similarities = np.dot(embedding_matrix, embedding_matrix.T)
        
        return similarities
