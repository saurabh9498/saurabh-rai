"""
Feature Store for Real-Time Fraud Detection

Redis-backed feature store for sub-millisecond feature retrieval
and real-time velocity calculations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    velocity_windows: List[str] = field(default_factory=lambda: ["1h", "6h", "24h", "7d"])
    aggregation_windows: List[str] = field(default_factory=lambda: ["1h", "24h", "7d", "30d"])
    ttl_days: int = 90
    
    # Window durations in seconds
    @property
    def window_seconds(self) -> Dict[str, int]:
        return {
            "1h": 3600,
            "6h": 21600,
            "24h": 86400,
            "7d": 604800,
            "30d": 2592000,
        }


@dataclass
class TransactionFeatures:
    """Features computed for a transaction."""
    card_id: str
    timestamp: datetime
    
    # Velocity features
    txn_count_1h: int = 0
    txn_count_6h: int = 0
    txn_count_24h: int = 0
    txn_count_7d: int = 0
    
    # Amount aggregations
    amount_sum_1h: float = 0.0
    amount_sum_24h: float = 0.0
    amount_avg_30d: float = 0.0
    amount_std_30d: float = 0.0
    
    # Behavioral features
    time_since_last_txn: float = 0.0  # seconds
    unique_merchants_24h: int = 0
    unique_channels_24h: int = 0
    
    # Pattern features
    is_first_transaction: bool = False
    is_new_merchant: bool = False
    is_new_device: bool = False
    deviation_from_avg: float = 0.0
    
    # Risk indicators
    merchant_risk_score: float = 0.0
    device_risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model input."""
        return {
            "txn_count_1h": self.txn_count_1h,
            "txn_count_6h": self.txn_count_6h,
            "txn_count_24h": self.txn_count_24h,
            "txn_count_7d": self.txn_count_7d,
            "amount_sum_1h": self.amount_sum_1h,
            "amount_sum_24h": self.amount_sum_24h,
            "amount_avg_30d": self.amount_avg_30d,
            "amount_std_30d": self.amount_std_30d,
            "time_since_last_txn": self.time_since_last_txn,
            "unique_merchants_24h": self.unique_merchants_24h,
            "unique_channels_24h": self.unique_channels_24h,
            "is_first_transaction": int(self.is_first_transaction),
            "is_new_merchant": int(self.is_new_merchant),
            "is_new_device": int(self.is_new_device),
            "deviation_from_avg": self.deviation_from_avg,
            "merchant_risk_score": self.merchant_risk_score,
            "device_risk_score": self.device_risk_score,
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array(list(self.to_dict().values()), dtype=np.float32)


class FeatureStore:
    """
    Redis-backed feature store for real-time feature computation.
    
    Supports:
    - Velocity tracking (transaction counts per time window)
    - Amount aggregations (sum, avg, std per window)
    - Behavioral features (unique merchants, channels, etc.)
    - Risk scores (merchant, device, card)
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: Optional[FeatureConfig] = None,
    ):
        self.redis_url = redis_url
        self.config = config or FeatureConfig()
        self._redis: Optional[redis.Redis] = None
        
    async def connect(self):
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using mock feature store")
            return
            
        self._redis = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self._redis.ping()
        logger.info(f"Connected to Redis: {self.redis_url}")
        
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            
    async def get_features(
        self,
        card_id: str,
        merchant_id: str,
        device_id: str,
        amount: float,
        timestamp: datetime,
    ) -> TransactionFeatures:
        """
        Get all features for a transaction.
        
        This is the main entry point for feature retrieval during scoring.
        Computes all features in parallel for minimum latency.
        """
        features = TransactionFeatures(card_id=card_id, timestamp=timestamp)
        
        if not self._redis:
            return features
            
        # Run all feature computations in parallel
        velocity_task = self._get_velocity_features(card_id, timestamp)
        amount_task = self._get_amount_features(card_id, timestamp)
        behavioral_task = self._get_behavioral_features(card_id, merchant_id, device_id, timestamp)
        risk_task = self._get_risk_scores(merchant_id, device_id)
        
        velocity, amounts, behavioral, risks = await asyncio.gather(
            velocity_task, amount_task, behavioral_task, risk_task
        )
        
        # Velocity features
        features.txn_count_1h = velocity.get("1h", 0)
        features.txn_count_6h = velocity.get("6h", 0)
        features.txn_count_24h = velocity.get("24h", 0)
        features.txn_count_7d = velocity.get("7d", 0)
        
        # Amount features
        features.amount_sum_1h = amounts.get("sum_1h", 0.0)
        features.amount_sum_24h = amounts.get("sum_24h", 0.0)
        features.amount_avg_30d = amounts.get("avg_30d", 0.0)
        features.amount_std_30d = amounts.get("std_30d", 0.0)
        
        # Behavioral features
        features.time_since_last_txn = behavioral.get("time_since_last", 0.0)
        features.unique_merchants_24h = behavioral.get("unique_merchants", 0)
        features.unique_channels_24h = behavioral.get("unique_channels", 0)
        features.is_first_transaction = behavioral.get("is_first", False)
        features.is_new_merchant = behavioral.get("is_new_merchant", False)
        features.is_new_device = behavioral.get("is_new_device", False)
        
        # Deviation from average
        if features.amount_std_30d > 0:
            features.deviation_from_avg = (amount - features.amount_avg_30d) / features.amount_std_30d
        
        # Risk scores
        features.merchant_risk_score = risks.get("merchant", 0.0)
        features.device_risk_score = risks.get("device", 0.0)
        
        return features
    
    async def _get_velocity_features(
        self,
        card_id: str,
        timestamp: datetime,
    ) -> Dict[str, int]:
        """Get transaction velocity for each time window."""
        results = {}
        
        for window, seconds in self.config.window_seconds.items():
            key = f"velocity:{card_id}:{window}"
            
            # Use Redis sorted set with timestamp as score
            min_time = timestamp.timestamp() - seconds
            count = await self._redis.zcount(key, min_time, "+inf")
            results[window] = count
            
        return results
    
    async def _get_amount_features(
        self,
        card_id: str,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Get amount aggregations."""
        results = {}
        
        # Get amount history
        key = f"amounts:{card_id}"
        
        # Sum for 1h
        min_1h = timestamp.timestamp() - 3600
        amounts_1h = await self._redis.zrangebyscore(key, min_1h, "+inf")
        results["sum_1h"] = sum(float(a) for a in amounts_1h)
        
        # Sum for 24h
        min_24h = timestamp.timestamp() - 86400
        amounts_24h = await self._redis.zrangebyscore(key, min_24h, "+inf")
        results["sum_24h"] = sum(float(a) for a in amounts_24h)
        
        # Avg and std for 30d
        min_30d = timestamp.timestamp() - 2592000
        amounts_30d = await self._redis.zrangebyscore(key, min_30d, "+inf")
        amounts_30d = [float(a) for a in amounts_30d]
        
        if amounts_30d:
            results["avg_30d"] = np.mean(amounts_30d)
            results["std_30d"] = np.std(amounts_30d) if len(amounts_30d) > 1 else 0.0
        else:
            results["avg_30d"] = 0.0
            results["std_30d"] = 0.0
            
        return results
    
    async def _get_behavioral_features(
        self,
        card_id: str,
        merchant_id: str,
        device_id: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Get behavioral features."""
        results = {}
        
        # Time since last transaction
        last_txn_key = f"last_txn:{card_id}"
        last_txn = await self._redis.get(last_txn_key)
        if last_txn:
            results["time_since_last"] = timestamp.timestamp() - float(last_txn)
            results["is_first"] = False
        else:
            results["time_since_last"] = 0.0
            results["is_first"] = True
            
        # Unique merchants in 24h
        merchants_key = f"merchants:{card_id}:24h"
        results["unique_merchants"] = await self._redis.scard(merchants_key)
        
        # Unique channels in 24h
        channels_key = f"channels:{card_id}:24h"
        results["unique_channels"] = await self._redis.scard(channels_key)
        
        # Is new merchant for this card
        known_merchants_key = f"known_merchants:{card_id}"
        results["is_new_merchant"] = not await self._redis.sismember(known_merchants_key, merchant_id)
        
        # Is new device for this card
        known_devices_key = f"known_devices:{card_id}"
        results["is_new_device"] = not await self._redis.sismember(known_devices_key, device_id)
        
        return results
    
    async def _get_risk_scores(
        self,
        merchant_id: str,
        device_id: str,
    ) -> Dict[str, float]:
        """Get pre-computed risk scores."""
        results = {}
        
        # Merchant risk score
        merchant_risk = await self._redis.get(f"risk:merchant:{merchant_id}")
        results["merchant"] = float(merchant_risk) if merchant_risk else 0.0
        
        # Device risk score
        device_risk = await self._redis.get(f"risk:device:{device_id}")
        results["device"] = float(device_risk) if device_risk else 0.0
        
        return results
    
    async def update_features(
        self,
        card_id: str,
        merchant_id: str,
        device_id: str,
        channel: str,
        amount: float,
        timestamp: datetime,
    ):
        """
        Update feature store after a transaction.
        
        Called after scoring to maintain feature freshness.
        """
        if not self._redis:
            return
            
        ts = timestamp.timestamp()
        ttl = self.config.ttl_days * 86400
        
        pipe = self._redis.pipeline()
        
        # Update velocity (sorted sets)
        for window in self.config.velocity_windows:
            key = f"velocity:{card_id}:{window}"
            pipe.zadd(key, {str(ts): ts})
            pipe.expire(key, ttl)
            
        # Update amounts
        amounts_key = f"amounts:{card_id}"
        pipe.zadd(amounts_key, {str(amount): ts})
        pipe.expire(amounts_key, ttl)
        
        # Update last transaction time
        pipe.set(f"last_txn:{card_id}", str(ts), ex=ttl)
        
        # Update merchants (with 24h expiry)
        pipe.sadd(f"merchants:{card_id}:24h", merchant_id)
        pipe.expire(f"merchants:{card_id}:24h", 86400)
        
        # Update known merchants
        pipe.sadd(f"known_merchants:{card_id}", merchant_id)
        pipe.expire(f"known_merchants:{card_id}", ttl)
        
        # Update channels
        pipe.sadd(f"channels:{card_id}:24h", channel)
        pipe.expire(f"channels:{card_id}:24h", 86400)
        
        # Update known devices
        pipe.sadd(f"known_devices:{card_id}", device_id)
        pipe.expire(f"known_devices:{card_id}", ttl)
        
        await pipe.execute()
        
    async def cleanup_expired(self, card_id: str, timestamp: datetime):
        """Remove expired entries from sorted sets."""
        if not self._redis:
            return
            
        pipe = self._redis.pipeline()
        
        for window, seconds in self.config.window_seconds.items():
            key = f"velocity:{card_id}:{window}"
            min_time = timestamp.timestamp() - seconds
            pipe.zremrangebyscore(key, "-inf", min_time)
            
        # Cleanup amounts older than 30d
        amounts_key = f"amounts:{card_id}"
        min_30d = timestamp.timestamp() - 2592000
        pipe.zremrangebyscore(amounts_key, "-inf", min_30d)
        
        await pipe.execute()


class MockFeatureStore(FeatureStore):
    """Mock feature store for testing without Redis."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config=config)
        self._data: Dict[str, Any] = {}
        
    async def connect(self):
        logger.info("Using mock feature store")
        
    async def close(self):
        pass
        
    async def get_features(
        self,
        card_id: str,
        merchant_id: str,
        device_id: str,
        amount: float,
        timestamp: datetime,
    ) -> TransactionFeatures:
        """Return random features for testing."""
        features = TransactionFeatures(card_id=card_id, timestamp=timestamp)
        
        # Generate realistic test features
        features.txn_count_1h = np.random.randint(0, 5)
        features.txn_count_6h = np.random.randint(0, 15)
        features.txn_count_24h = np.random.randint(0, 30)
        features.txn_count_7d = np.random.randint(0, 100)
        
        features.amount_sum_1h = np.random.uniform(0, 500)
        features.amount_sum_24h = np.random.uniform(0, 2000)
        features.amount_avg_30d = np.random.uniform(50, 200)
        features.amount_std_30d = np.random.uniform(20, 100)
        
        features.time_since_last_txn = np.random.uniform(60, 86400)
        features.unique_merchants_24h = np.random.randint(1, 10)
        features.unique_channels_24h = np.random.randint(1, 3)
        
        features.is_first_transaction = np.random.random() < 0.01
        features.is_new_merchant = np.random.random() < 0.1
        features.is_new_device = np.random.random() < 0.05
        
        if features.amount_std_30d > 0:
            features.deviation_from_avg = (amount - features.amount_avg_30d) / features.amount_std_30d
            
        features.merchant_risk_score = np.random.uniform(0, 0.3)
        features.device_risk_score = np.random.uniform(0, 0.2)
        
        return features
        
    async def update_features(self, *args, **kwargs):
        pass
