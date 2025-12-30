"""
API Schemas

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class TransactionRequest(BaseModel):
    """Request schema for scoring a transaction."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    card_id: str = Field(..., description="Card/account identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(default="unknown", description="Merchant category code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Transaction timestamp")
    channel: str = Field(default="online", description="Transaction channel (online/pos/atm)")
    ip_address: str = Field(default="", description="Client IP address")
    device_id: str = Field(default="", description="Device identifier")
    location: Optional[Dict[str, float]] = Field(default=None, description="Lat/lon coordinates")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_abc123",
                "card_id": "card_xyz789",
                "amount": 150.00,
                "merchant_id": "merch_001",
                "merchant_category": "retail",
                "timestamp": "2024-01-15T10:30:00Z",
                "channel": "online",
                "ip_address": "192.168.1.1",
                "device_id": "device_123",
            }
        }
        
    @validator("channel")
    def validate_channel(cls, v):
        valid_channels = ["online", "pos", "atm", "mobile", "phone"]
        if v.lower() not in valid_channels:
            v = "unknown"
        return v.lower()


class TransactionResponse(BaseModel):
    """Response schema for transaction scoring."""
    
    transaction_id: str = Field(..., description="Transaction identifier")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    decision: str = Field(..., description="Decision (approve/review/decline/step_up)")
    
    # Individual model scores
    xgboost_score: float = Field(default=0.0, description="XGBoost model score")
    neural_net_score: float = Field(default=0.0, description="Neural network score")
    isolation_forest_score: float = Field(default=0.0, description="Anomaly score")
    
    # Explanation
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    
    # Timing
    latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_abc123",
                "risk_score": 0.15,
                "decision": "approve",
                "xgboost_score": 0.12,
                "neural_net_score": 0.18,
                "isolation_forest_score": 0.10,
                "risk_factors": [],
                "latency_ms": 5.2,
            }
        }


class BatchRequest(BaseModel):
    """Request schema for batch scoring."""
    
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "txn_001",
                        "card_id": "card_abc",
                        "amount": 100.00,
                        "merchant_id": "merch_001",
                    },
                    {
                        "transaction_id": "txn_002",
                        "card_id": "card_xyz",
                        "amount": 250.00,
                        "merchant_id": "merch_002",
                    },
                ]
            }
        }


class BatchResponse(BaseModel):
    """Response schema for batch scoring."""
    
    results: List[TransactionResponse] = Field(..., description="Scoring results")
    total_latency_ms: float = Field(..., description="Total processing time")
    avg_latency_ms: float = Field(..., description="Average per-transaction latency")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Current timestamp")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    version: str = Field(..., description="API version")


class ModelInfo(BaseModel):
    """Response schema for model information."""
    
    model_type: str = Field(..., description="Model type (ensemble)")
    version: str = Field(..., description="Model version")
    models: List[str] = Field(..., description="Component models")
    weights: Dict[str, float] = Field(..., description="Model weights")
    thresholds: Dict[str, float] = Field(..., description="Decision thresholds")


class RulesRequest(BaseModel):
    """Request schema for rules evaluation."""
    
    transaction: TransactionRequest
    rules: Optional[List[str]] = Field(default=None, description="Specific rules to evaluate")


class RulesResponse(BaseModel):
    """Response schema for rules evaluation."""
    
    transaction_id: str
    triggered_rules: List[Dict[str, Any]] = Field(default_factory=list)
    action: str = Field(default="none")


class FeedbackRequest(BaseModel):
    """Request schema for providing fraud feedback."""
    
    transaction_id: str = Field(..., description="Transaction identifier")
    is_fraud: bool = Field(..., description="Whether transaction was fraud")
    feedback_source: str = Field(default="manual", description="Source of feedback")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class FeedbackResponse(BaseModel):
    """Response schema for feedback acknowledgment."""
    
    transaction_id: str
    received: bool = True
    message: str = "Feedback recorded"


class MetricsResponse(BaseModel):
    """Response schema for system metrics."""
    
    requests_per_second: float
    avg_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    fraud_rate: float
    false_positive_rate: float


class DriftAlert(BaseModel):
    """Schema for model drift alerts."""
    
    alert_id: str
    feature: str
    drift_score: float
    threshold: float
    timestamp: datetime
    severity: str = Field(default="warning")
