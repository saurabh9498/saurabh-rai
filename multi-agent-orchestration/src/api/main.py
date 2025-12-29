"""
FastAPI Application for Multi-Agent AI System.

Provides REST API endpoints for interacting with the
multi-agent system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from .routes import router
from .schemas import HealthResponse
from ..core.config import settings


logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Multi-Agent AI System API...")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize any required resources here
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Agent AI System API...")


# Create FastAPI app
app = FastAPI(
    title="Multi-Agent AI System",
    description="""
    Enterprise-grade multi-agent orchestration system with RAG capabilities.
    
    ## Features
    
    * **Multi-Agent Orchestration**: Coordinate specialized AI agents for complex tasks
    * **RAG Pipeline**: Retrieve and generate from your knowledge base
    * **Tool Use**: Agents can use tools like web search, database queries, and code execution
    * **Streaming**: Real-time streaming responses
    
    ## Agents
    
    * **Research Agent**: Information retrieval and synthesis
    * **Analyst Agent**: Data analysis and visualization
    * **Code Agent**: Code generation and review
    * **Orchestrator**: Coordinates multi-agent workflows
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request, call_next):
    """Add response timing header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs"
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.version,
        environment=settings.environment
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug
    )
