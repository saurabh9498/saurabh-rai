"""
User Feature Computation

Computes and manages user-level features for recommendations:
- Demographic features
- Behavioral aggregations
- Real-time session features
- User embeddings
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with computed features."""
    user_id: str
    
    # Demographics
    age_bucket: int = 0
    gender: int = 0
    country: int = 0
    
    # Account stats
    account_age_days: float = 0.0
    total_orders: int = 0
    total_spend: float = 0.0
    
    # Engagement metrics
    click_rate_7d: float = 0.0
    purchase_rate_7d: float = 0.0
    session_count_30d: int = 0
    avg_session_duration: float = 0.0
    
    # Preferences
    preferred_categories: List[int] = None
    preferred_brands: List[int] = None
    preferred_price_range: tuple = None
    
    # History
    recent_views: List[str] = None
    recent_purchases: List[str] = None
    recent_searches: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'age_bucket': self.age_bucket,
            'gender': self.gender,
            'country': self.country,
            'account_age_days': self.account_age_days,
            'total_orders': self.total_orders,
            'total_spend': self.total_spend,
            'click_rate_7d': self.click_rate_7d,
            'purchase_rate_7d': self.purchase_rate_7d,
            'session_count_30d': self.session_count_30d,
            'avg_session_duration': self.avg_session_duration,
            'preferred_categories': self.preferred_categories or [],
            'preferred_brands': self.preferred_brands or [],
            'recent_views': self.recent_views or [],
            'recent_purchases': self.recent_purchases or [],
        }


class UserFeatureComputer:
    """Computes user features from raw data."""
    
    def __init__(
        self,
        lookback_days: int = 30,
        max_history_length: int = 50,
    ):
        self.lookback_days = lookback_days
        self.max_history_length = max_history_length
    
    def compute_features(
        self,
        user_id: str,
        user_data: Dict[str, Any],
        interactions: List[Dict[str, Any]],
    ) -> UserProfile:
        """
        Compute all user features.
        
        Args:
            user_id: User identifier
            user_data: Static user attributes
            interactions: User's interaction history
            
        Returns:
            UserProfile with computed features
        """
        profile = UserProfile(user_id=user_id)
        
        # Demographics
        profile.age_bucket = self._compute_age_bucket(user_data.get('birthdate'))
        profile.gender = user_data.get('gender_id', 0)
        profile.country = user_data.get('country_id', 0)
        
        # Account stats
        profile.account_age_days = self._compute_account_age(
            user_data.get('created_at')
        )
        profile.total_orders = user_data.get('total_orders', 0)
        profile.total_spend = user_data.get('total_spend', 0.0)
        
        # Behavioral metrics
        if interactions:
            profile.click_rate_7d = self._compute_click_rate(interactions, days=7)
            profile.purchase_rate_7d = self._compute_purchase_rate(interactions, days=7)
            profile.session_count_30d = self._compute_session_count(interactions, days=30)
            
            # History
            profile.recent_views = self._get_recent_items(
                interactions, event_type='view'
            )
            profile.recent_purchases = self._get_recent_items(
                interactions, event_type='purchase'
            )
            
            # Preferences
            profile.preferred_categories = self._compute_preferred_categories(
                interactions
            )
            profile.preferred_brands = self._compute_preferred_brands(interactions)
        
        return profile
    
    def _compute_age_bucket(self, birthdate: Optional[str]) -> int:
        """Convert birthdate to age bucket."""
        if not birthdate:
            return 0
        
        try:
            birth = datetime.fromisoformat(birthdate)
            age = (datetime.now() - birth).days // 365
            
            buckets = [18, 25, 35, 45, 55, 65, 75, 100]
            for i, boundary in enumerate(buckets):
                if age < boundary:
                    return i + 1
            return len(buckets)
        except:
            return 0
    
    def _compute_account_age(self, created_at: Optional[str]) -> float:
        """Compute account age in days."""
        if not created_at:
            return 0.0
        
        try:
            created = datetime.fromisoformat(created_at)
            return (datetime.now() - created).days
        except:
            return 0.0
    
    def _compute_click_rate(
        self,
        interactions: List[Dict],
        days: int = 7,
    ) -> float:
        """Compute click-through rate over recent period."""
        cutoff = datetime.now() - timedelta(days=days)
        
        impressions = 0
        clicks = 0
        
        for interaction in interactions:
            timestamp = interaction.get('timestamp')
            if timestamp and datetime.fromisoformat(timestamp) > cutoff:
                event_type = interaction.get('event_type')
                if event_type == 'impression':
                    impressions += 1
                elif event_type == 'click':
                    clicks += 1
        
        return clicks / max(impressions, 1)
    
    def _compute_purchase_rate(
        self,
        interactions: List[Dict],
        days: int = 7,
    ) -> float:
        """Compute purchase rate over recent period."""
        cutoff = datetime.now() - timedelta(days=days)
        
        clicks = 0
        purchases = 0
        
        for interaction in interactions:
            timestamp = interaction.get('timestamp')
            if timestamp and datetime.fromisoformat(timestamp) > cutoff:
                event_type = interaction.get('event_type')
                if event_type == 'click':
                    clicks += 1
                elif event_type == 'purchase':
                    purchases += 1
        
        return purchases / max(clicks, 1)
    
    def _compute_session_count(
        self,
        interactions: List[Dict],
        days: int = 30,
    ) -> int:
        """Count unique sessions in recent period."""
        cutoff = datetime.now() - timedelta(days=days)
        sessions = set()
        
        for interaction in interactions:
            timestamp = interaction.get('timestamp')
            session_id = interaction.get('session_id')
            if timestamp and datetime.fromisoformat(timestamp) > cutoff:
                if session_id:
                    sessions.add(session_id)
        
        return len(sessions)
    
    def _get_recent_items(
        self,
        interactions: List[Dict],
        event_type: str,
    ) -> List[str]:
        """Get most recent items for given event type."""
        items = []
        
        for interaction in sorted(
            interactions,
            key=lambda x: x.get('timestamp', ''),
            reverse=True,
        ):
            if interaction.get('event_type') == event_type:
                item_id = interaction.get('item_id')
                if item_id and item_id not in items:
                    items.append(item_id)
                    if len(items) >= self.max_history_length:
                        break
        
        return items
    
    def _compute_preferred_categories(
        self,
        interactions: List[Dict],
        top_k: int = 5,
    ) -> List[int]:
        """Compute top preferred categories based on engagement."""
        category_scores = {}
        
        for interaction in interactions:
            category = interaction.get('category_id')
            if category:
                event_type = interaction.get('event_type')
                weight = {
                    'view': 1,
                    'click': 2,
                    'add_to_cart': 3,
                    'purchase': 5,
                }.get(event_type, 1)
                
                category_scores[category] = category_scores.get(category, 0) + weight
        
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [cat for cat, _ in sorted_categories[:top_k]]
    
    def _compute_preferred_brands(
        self,
        interactions: List[Dict],
        top_k: int = 5,
    ) -> List[int]:
        """Compute top preferred brands."""
        brand_scores = {}
        
        for interaction in interactions:
            brand = interaction.get('brand_id')
            if brand:
                event_type = interaction.get('event_type')
                weight = {'view': 1, 'click': 2, 'purchase': 5}.get(event_type, 1)
                brand_scores[brand] = brand_scores.get(brand, 0) + weight
        
        sorted_brands = sorted(
            brand_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [brand for brand, _ in sorted_brands[:top_k]]


class RealTimeUserFeatures:
    """Compute real-time session features."""
    
    def __init__(self):
        self.session_cache = {}
    
    def update_session(
        self,
        user_id: str,
        session_id: str,
        event: Dict[str, Any],
    ):
        """Update session state with new event."""
        key = f"{user_id}:{session_id}"
        
        if key not in self.session_cache:
            self.session_cache[key] = {
                'start_time': datetime.now(),
                'views': [],
                'clicks': [],
                'cart_items': [],
                'search_queries': [],
            }
        
        session = self.session_cache[key]
        event_type = event.get('event_type')
        
        if event_type == 'view':
            session['views'].append(event.get('item_id'))
        elif event_type == 'click':
            session['clicks'].append(event.get('item_id'))
        elif event_type == 'add_to_cart':
            session['cart_items'].append(event.get('item_id'))
        elif event_type == 'search':
            session['search_queries'].append(event.get('query'))
    
    def get_session_features(
        self,
        user_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get current session features."""
        key = f"{user_id}:{session_id}"
        session = self.session_cache.get(key, {})
        
        if not session:
            return {
                'session_duration_sec': 0,
                'pages_viewed': 0,
                'items_clicked': 0,
                'items_in_cart': 0,
                'recent_views': [],
            }
        
        duration = (datetime.now() - session.get('start_time', datetime.now())).seconds
        
        return {
            'session_duration_sec': duration,
            'pages_viewed': len(session.get('views', [])),
            'items_clicked': len(session.get('clicks', [])),
            'items_in_cart': len(session.get('cart_items', [])),
            'recent_views': session.get('views', [])[-10:],
            'recent_clicks': session.get('clicks', [])[-5:],
            'search_queries': session.get('search_queries', [])[-3:],
        }
