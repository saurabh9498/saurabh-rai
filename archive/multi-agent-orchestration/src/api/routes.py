"""
API Routes for Multi-Agent AI System.

Defines all REST API endpoints for the system.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse
from typing import List, Optional
import asyncio
import uuid
import time
import logging

from .schemas import (
    QueryRequest, QueryResponse, AgentTrace, TaskStatus,
    IngestRequest, IngestResponse,
    AgentsResponse, AgentInfo, AgentType,
    RAGQueryRequest, RAGQueryResponse, RAGResult,
    FeedbackRequest, FeedbackResponse,
    SystemStatus, ToolsResponse, ToolInfo
)
from ..agents import OrchestratorAgent
from ..rag import RAGPipeline
from ..tools import registry
from ..core.context import context_manager, ConversationContext


logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (would be properly initialized in production)
orchestrator: Optional[OrchestratorAgent] = None
rag_pipeline: Optional[RAGPipeline] = None
start_time = time.time()


def get_orchestrator() -> OrchestratorAgent:
    """Get or create orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        orchestrator = OrchestratorAgent()
    return orchestrator


# ==================== Query Endpoints ====================

@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Execute a multi-agent query",
    description="Execute a query using the multi-agent system"
)
async def execute_query(request: QueryRequest) -> QueryResponse:
    """Execute a query through the multi-agent system."""
    query_id = f"query_{uuid.uuid4().hex[:12]}"
    start = time.time()
    
    try:
        orch = get_orchestrator()
        context = context_manager.create_context(query_id)
        
        # Execute the query
        result = await orch.execute_workflow(
            task=request.query,
            context=context
        )
        
        execution_time = time.time() - start
        
        # Build agent traces
        traces = []
        for agent_name, agent_result in result.agent_results.items():
            traces.append(AgentTrace(
                agent=agent_name,
                actions=agent_result.get("actions_taken", []),
                duration=agent_result.get("execution_time", 0),
                status=TaskStatus.COMPLETED if agent_result.get("success") else TaskStatus.FAILED
            ))
        
        return QueryResponse(
            id=query_id,
            status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            execution_time=execution_time,
            result={
                "summary": result.final_output,
                "confidence": 0.85  # Placeholder
            },
            agent_traces=traces,
            error=result.errors[0] if result.errors else None
        )
        
    except Exception as e:
        logger.exception(f"Query execution failed: {e}")
        return QueryResponse(
            id=query_id,
            status=TaskStatus.FAILED,
            execution_time=time.time() - start,
            error=str(e)
        )


@router.post(
    "/query/stream",
    tags=["Query"],
    summary="Execute a streaming query",
    description="Execute a query with streaming response"
)
async def execute_streaming_query(request: QueryRequest):
    """Execute a query with streaming response."""
    async def generate():
        query_id = f"query_{uuid.uuid4().hex[:12]}"
        
        yield f"data: {{\"id\": \"{query_id}\", \"status\": \"started\"}}\n\n"
        
        try:
            orch = get_orchestrator()
            context = context_manager.create_context(query_id)
            
            # Simulate streaming (in production, would stream actual tokens)
            result = await orch.execute_workflow(
                task=request.query,
                context=context
            )
            
            # Stream the result
            if result.success:
                output = result.final_output
                # Stream in chunks
                chunk_size = 50
                for i in range(0, len(output), chunk_size):
                    chunk = output[i:i + chunk_size]
                    yield f"data: {{\"chunk\": \"{chunk}\"}}\n\n"
                    await asyncio.sleep(0.05)
            
            yield f"data: {{\"status\": \"completed\"}}\n\n"
            
        except Exception as e:
            yield f"data: {{\"status\": \"failed\", \"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# ==================== Ingest Endpoints ====================

@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingest"],
    summary="Ingest documents",
    description="Ingest documents into the knowledge base"
)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Ingest documents into the RAG pipeline."""
    try:
        # In production, would use actual RAG pipeline
        # For now, return mock response
        chunks_per_doc = 5  # Estimate
        
        return IngestResponse(
            success=True,
            documents_processed=len(request.documents),
            chunks_created=len(request.documents) * chunks_per_doc,
            collection=request.collection or "default"
        )
        
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        return IngestResponse(
            success=False,
            documents_processed=0,
            chunks_created=0,
            collection=request.collection or "default",
            error=str(e)
        )


# ==================== Agent Endpoints ====================

@router.get(
    "/agents",
    response_model=AgentsResponse,
    tags=["Agents"],
    summary="List available agents",
    description="Get information about all available agents"
)
async def list_agents() -> AgentsResponse:
    """List all available agents."""
    orch = get_orchestrator()
    
    agents = []
    for name, agent in orch.agents.items():
        agents.append(AgentInfo(
            name=name,
            type=AgentType(name) if name in [t.value for t in AgentType] else AgentType.RESEARCH,
            description=agent.description,
            tools=list(agent.tools.keys()),
            status="active"
        ))
    
    return AgentsResponse(agents=agents)


@router.get(
    "/agents/{agent_name}",
    response_model=AgentInfo,
    tags=["Agents"],
    summary="Get agent details",
    description="Get detailed information about a specific agent"
)
async def get_agent(
    agent_name: str = Path(..., description="Agent name")
) -> AgentInfo:
    """Get information about a specific agent."""
    orch = get_orchestrator()
    
    if agent_name not in orch.agents:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")
    
    agent = orch.agents[agent_name]
    
    return AgentInfo(
        name=agent_name,
        type=AgentType(agent_name) if agent_name in [t.value for t in AgentType] else AgentType.RESEARCH,
        description=agent.description,
        tools=list(agent.tools.keys()),
        status="active"
    )


# ==================== RAG Endpoints ====================

@router.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    tags=["RAG"],
    summary="Query the knowledge base",
    description="Execute a RAG query against the knowledge base"
)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    """Execute a RAG query."""
    start = time.time()
    
    # In production, would use actual RAG pipeline
    # Return mock response for now
    results = [
        RAGResult(
            content="Sample retrieved content for the query...",
            score=0.92,
            metadata={"source": "document1.pdf", "page": 5}
        ),
        RAGResult(
            content="Another relevant document section...",
            score=0.87,
            metadata={"source": "document2.pdf", "page": 12}
        )
    ]
    
    return RAGQueryResponse(
        query=request.query,
        results=results[:request.top_k],
        context="Combined context from retrieved documents...",
        execution_time=time.time() - start
    )


# ==================== Feedback Endpoints ====================

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    summary="Submit feedback",
    description="Submit feedback for a query execution"
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit feedback for a query."""
    # In production, would store feedback in database
    logger.info(f"Feedback received for query {request.query_id}: {request.rating}/5")
    
    return FeedbackResponse(
        success=True,
        message="Feedback received successfully"
    )


# ==================== System Endpoints ====================

@router.get(
    "/status",
    response_model=SystemStatus,
    tags=["System"],
    summary="Get system status",
    description="Get current system status and metrics"
)
async def get_status() -> SystemStatus:
    """Get system status."""
    return SystemStatus(
        status="healthy",
        uptime=time.time() - start_time,
        active_queries=0,  # Would track in production
        vector_store_status="connected",
        llm_status="connected"
    )


@router.get(
    "/tools",
    response_model=ToolsResponse,
    tags=["Tools"],
    summary="List available tools",
    description="Get information about all available tools"
)
async def list_tools() -> ToolsResponse:
    """List all available tools."""
    tools = []
    
    for tool_def in registry.list_tools():
        tools.append(ToolInfo(
            name=tool_def.name,
            description=tool_def.description,
            category=tool_def.category.value,
            parameters=tool_def.parameters,
            enabled=tool_def.enabled
        ))
    
    return ToolsResponse(tools=tools)
