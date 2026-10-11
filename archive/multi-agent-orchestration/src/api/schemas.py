"""
Pydantic Schemas for API.

Defines request and response models for the API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class AgentType(str, Enum):
    """Available agent types."""
    RESEARCH = "research"
    ANALYST = "analyst"
    CODE = "code"
    ORCHESTRATOR = "orchestrator"


class OutputFormat(str, Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    DETAILED = "detailed"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== Health ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== Query ====================

class QueryRequest(BaseModel):
    """Request for executing a query."""
    query: str = Field(..., description="The query to execute", min_length=1)
    agents: Optional[List[AgentType]] = Field(
        default=None,
        description="Specific agents to use (defaults to auto-selection)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for the query"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.TEXT,
        description="Desired output format"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the key trends in our Q3 support tickets?",
                "agents": ["research", "analyst"],
                "context": {
                    "time_range": "last_30_days",
                    "department": "support"
                },
                "output_format": "detailed"
            }
        }


class AgentTrace(BaseModel):
    """Trace of an agent's execution."""
    agent: str
    actions: List[Dict[str, Any]]
    duration: float
    status: TaskStatus


class QueryResponse(BaseModel):
    """Response from a query execution."""
    id: str = Field(..., description="Unique query ID")
    status: TaskStatus
    execution_time: float = Field(..., description="Total execution time in seconds")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query results"
    )
    agent_traces: List[AgentTrace] = Field(
        default_factory=list,
        description="Traces of agent executions"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "query_abc123",
                "status": "completed",
                "execution_time": 4.2,
                "result": {
                    "summary": "Analysis of support tickets reveals...",
                    "confidence": 0.92
                },
                "agent_traces": [
                    {
                        "agent": "research",
                        "actions": [],
                        "duration": 1.8,
                        "status": "completed"
                    }
                ]
            }
        }


# ==================== Ingest ====================

class IngestRequest(BaseModel):
    """Request for ingesting documents."""
    documents: List[str] = Field(
        ...,
        description="List of document texts to ingest",
        min_items=1
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata to attach to all documents"
    )
    collection: Optional[str] = Field(
        default="default",
        description="Collection to ingest into"
    )
    chunking_strategy: Optional[str] = Field(
        default="recursive",
        description="Chunking strategy to use"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    "This is the first document content...",
                    "This is the second document content..."
                ],
                "metadata": {
                    "source": "manual_upload",
                    "category": "documentation"
                },
                "collection": "knowledge_base"
            }
        }


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    success: bool
    documents_processed: int
    chunks_created: int
    collection: str
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "documents_processed": 2,
                "chunks_created": 15,
                "collection": "knowledge_base"
            }
        }


# ==================== Agents ====================

class AgentInfo(BaseModel):
    """Information about an agent."""
    name: str
    type: AgentType
    description: str
    tools: List[str]
    status: str


class AgentsResponse(BaseModel):
    """Response listing available agents."""
    agents: List[AgentInfo]


# ==================== RAG ====================

class RAGQueryRequest(BaseModel):
    """Request for RAG query."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results", ge=1, le=20)
    collection: Optional[str] = Field(default="default", description="Collection to search")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    use_reranking: bool = Field(default=True, description="Whether to use reranking")


class RAGResult(BaseModel):
    """A single RAG result."""
    content: str
    score: float
    metadata: Dict[str, Any]


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""
    query: str
    results: List[RAGResult]
    context: str
    execution_time: float


# ==================== Feedback ====================

class FeedbackRequest(BaseModel):
    """Request for submitting feedback."""
    query_id: str = Field(..., description="ID of the query to provide feedback for")
    rating: int = Field(..., description="Rating from 1-5", ge=1, le=5)
    feedback: Optional[str] = Field(default=None, description="Additional feedback text")
    improvements: Optional[List[str]] = Field(
        default=None,
        description="Suggested improvements"
    )


class FeedbackResponse(BaseModel):
    """Response from feedback submission."""
    success: bool
    message: str


# ==================== Status ====================

class SystemStatus(BaseModel):
    """System status information."""
    status: str
    uptime: float
    active_queries: int
    vector_store_status: str
    llm_status: str


# ==================== Tools ====================

class ToolInfo(BaseModel):
    """Information about a tool."""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    enabled: bool


class ToolsResponse(BaseModel):
    """Response listing available tools."""
    tools: List[ToolInfo]
