"""
Fraud Detection API

FastAPI application for real-time fraud scoring.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .schemas import (
    TransactionRequest,
    TransactionResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
)
from ..features.feature_store import FeatureStore, MockFeatureStore
from ..models.ensemble import FraudEnsemble, EnsembleConfig, Decision


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Prometheus metrics
REQUEST_COUNT = Counter(
    "fraud_requests_total",
    "Total fraud scoring requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
SCORE_HISTOGRAM = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
DECISION_COUNT = Counter(
    "fraud_decisions_total",
    "Total decisions by type",
    ["decision"],
)


# Global state
class AppState:
    feature_store: Optional[FeatureStore] = None
    ensemble: Optional[FraudEnsemble] = None
    start_time: datetime = datetime.utcnow()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Fraud Detection API...")
    
    # Initialize feature store
    try:
        state.feature_store = FeatureStore()
        await state.feature_store.connect()
    except Exception as e:
        logger.warning(f"Redis not available, using mock: {e}")
        state.feature_store = MockFeatureStore()
        await state.feature_store.connect()
        
    # Initialize ensemble (mock for demo)
    state.ensemble = create_mock_ensemble()
    
    logger.info("API initialization complete")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if state.feature_store:
        await state.feature_store.close()


def create_mock_ensemble() -> FraudEnsemble:
    """Create a mock ensemble for demo purposes."""
    from ..models.xgboost_model import MockXGBoostModel
    from ..models.neural_net import MockNeuralNetModel
    from ..models.isolation_forest import MockIsolationForestModel
    
    ensemble = FraudEnsemble()
    ensemble.xgboost = MockXGBoostModel()
    ensemble.neural_net = MockNeuralNetModel()
    ensemble.isolation_forest = MockIsolationForestModel()
    ensemble._is_trained = True
    
    return ensemble


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with ML ensemble scoring",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{duration * 1000:.2f}ms"
    return response


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime_seconds=(datetime.utcnow() - state.start_time).total_seconds(),
        version="1.0.0",
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if state.ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/v1/score", response_model=TransactionResponse)
async def score_transaction(request: TransactionRequest):
    """
    Score a single transaction for fraud.
    
    Returns risk score (0-1) and decision (approve/review/decline).
    """
    start_time = time.perf_counter()
    
    try:
        # Get features from feature store
        features = await state.feature_store.get_features(
            card_id=request.card_id,
            merchant_id=request.merchant_id,
            device_id=request.device_id,
            amount=request.amount,
            timestamp=request.timestamp,
        )
        
        # Score with ensemble
        result = state.ensemble.score(
            features.to_array(),
            transaction_id=request.transaction_id,
        )
        
        # Update feature store
        await state.feature_store.update_features(
            card_id=request.card_id,
            merchant_id=request.merchant_id,
            device_id=request.device_id,
            channel=request.channel,
            amount=request.amount,
            timestamp=request.timestamp,
        )
        
        # Record metrics
        latency = time.perf_counter() - start_time
        REQUEST_COUNT.labels(endpoint="score", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="score").observe(latency)
        SCORE_HISTOGRAM.observe(result.risk_score)
        DECISION_COUNT.labels(decision=result.decision.value).inc()
        
        return TransactionResponse(
            transaction_id=request.transaction_id,
            risk_score=result.risk_score,
            decision=result.decision.value,
            xgboost_score=result.xgboost_score,
            neural_net_score=result.neural_net_score,
            isolation_forest_score=result.isolation_forest_score,
            risk_factors=result.risk_factors,
            latency_ms=latency * 1000,
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="score", status="error").inc()
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/score/batch", response_model=BatchResponse)
async def score_batch(request: BatchRequest):
    """
    Score multiple transactions.
    
    Optimized for throughput with batch processing.
    """
    start_time = time.perf_counter()
    
    try:
        responses = []
        
        for txn in request.transactions:
            # Get features
            features = await state.feature_store.get_features(
                card_id=txn.card_id,
                merchant_id=txn.merchant_id,
                device_id=txn.device_id,
                amount=txn.amount,
                timestamp=txn.timestamp,
            )
            
            # Score
            result = state.ensemble.score(
                features.to_array(),
                transaction_id=txn.transaction_id,
            )
            
            responses.append(TransactionResponse(
                transaction_id=txn.transaction_id,
                risk_score=result.risk_score,
                decision=result.decision.value,
                xgboost_score=result.xgboost_score,
                neural_net_score=result.neural_net_score,
                isolation_forest_score=result.isolation_forest_score,
                risk_factors=result.risk_factors,
                latency_ms=result.latency_ms,
            ))
            
            # Record score metric
            SCORE_HISTOGRAM.observe(result.risk_score)
            DECISION_COUNT.labels(decision=result.decision.value).inc()
            
        latency = time.perf_counter() - start_time
        REQUEST_COUNT.labels(endpoint="batch", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="batch").observe(latency)
        
        return BatchResponse(
            results=responses,
            total_latency_ms=latency * 1000,
            avg_latency_ms=(latency * 1000) / len(request.transactions),
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="batch", status="error").inc()
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    return ModelInfo(
        model_type="ensemble",
        version="1.0.0",
        models=["xgboost", "neural_net", "isolation_forest"],
        weights={
            "xgboost": state.ensemble.config.xgboost_weight,
            "neural_net": state.ensemble.config.neural_net_weight,
            "isolation_forest": state.ensemble.config.isolation_forest_weight,
        },
        thresholds={
            "approve": state.ensemble.config.approve_threshold,
            "review": state.ensemble.config.review_threshold,
            "decline": state.ensemble.config.decline_threshold,
        },
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
