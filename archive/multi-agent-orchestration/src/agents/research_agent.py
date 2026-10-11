"""
Research Agent.

Specialized agent for information retrieval, web search,
RAG queries, and content summarization.
"""

from typing import List, Dict, Any, Optional
import json
import logging

from .base_agent import BaseAgent, Tool, AgentResult
from ..core.context import ConversationContext
from ..core.llm import get_llm


logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Agent specialized in research tasks including:
    - RAG-based document retrieval
    - Web search
    - Content summarization
    - Source synthesis
    """
    
    def __init__(
        self,
        name: str = "research_agent",
        rag_pipeline = None,
        web_search_enabled: bool = True,
        **kwargs
    ):
        # Initialize tools based on capabilities
        tools = []
        
        self.rag_pipeline = rag_pipeline
        self.web_search_enabled = web_search_enabled
        
        super().__init__(
            name=name,
            description="Research agent for information retrieval and synthesis",
            tools=tools,
            **kwargs
        )
        
        # Register tools after initialization
        self._register_default_tools()
    
    def _default_system_prompt(self) -> str:
        return """You are a Research Agent specialized in finding and synthesizing information.

Your capabilities:
1. RAG Query: Search the internal knowledge base for relevant documents
2. Web Search: Find current information from the web (when enabled)
3. Summarize: Create concise summaries of content
4. Synthesize: Combine information from multiple sources

Guidelines:
- Always cite your sources
- Distinguish between facts and interpretations
- Acknowledge uncertainty when information is incomplete
- Prioritize authoritative and recent sources
- Structure findings clearly

When given a research task:
1. First, understand what information is needed
2. Plan your research approach
3. Execute searches (RAG first, then web if needed)
4. Synthesize and summarize findings
5. Provide sources and confidence levels

Respond in a structured format with clear sections for findings, sources, and any caveats."""
    
    def _register_default_tools(self) -> None:
        """Register the default research tools."""
        
        # RAG Query Tool
        self.register_tool(Tool(
            name="rag_query",
            description="Search the internal knowledge base using semantic search",
            function=self._rag_query,
            parameters={
                "query": {"type": "string", "description": "The search query"},
                "top_k": {"type": "integer", "description": "Number of results to return", "default": 5}
            },
            required_params=["query"]
        ))
        
        # Summarize Tool
        self.register_tool(Tool(
            name="summarize",
            description="Summarize a piece of text",
            function=self._summarize,
            parameters={
                "text": {"type": "string", "description": "The text to summarize"},
                "max_length": {"type": "integer", "description": "Maximum summary length in words", "default": 200}
            },
            required_params=["text"]
        ))
        
        # Web Search Tool (if enabled)
        if self.web_search_enabled:
            self.register_tool(Tool(
                name="web_search",
                description="Search the web for current information",
                function=self._web_search,
                parameters={
                    "query": {"type": "string", "description": "The search query"},
                    "num_results": {"type": "integer", "description": "Number of results", "default": 5}
                },
                required_params=["query"]
            ))
    
    async def _execute_task(
        self,
        task: str,
        context: ConversationContext,
        use_rag: bool = True,
        use_web: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a research task.
        
        Args:
            task: The research question or task
            context: Conversation context
            use_rag: Whether to use RAG search
            use_web: Whether to use web search
            
        Returns:
            Research results with sources
        """
        results = {
            "query": task,
            "findings": [],
            "sources": [],
            "summary": "",
            "confidence": 0.0
        }
        
        # Step 1: Understand the query and plan approach
        plan = await self._plan_research(task)
        self._log_action("research_plan", {"plan": plan})
        
        # Step 2: Execute RAG search if enabled
        if use_rag and self.rag_pipeline:
            rag_results = await self._use_tool("rag_query", query=task, top_k=5)
            results["findings"].extend(rag_results.get("documents", []))
            results["sources"].extend(rag_results.get("sources", []))
        
        # Step 3: Execute web search if enabled and needed
        if use_web and self.web_search_enabled:
            web_results = await self._use_tool("web_search", query=task, num_results=5)
            results["findings"].extend(web_results.get("results", []))
            results["sources"].extend([r.get("url") for r in web_results.get("results", [])])
        
        # Step 4: Synthesize findings
        if results["findings"]:
            synthesis = await self._synthesize_findings(task, results["findings"])
            results["summary"] = synthesis["summary"]
            results["confidence"] = synthesis["confidence"]
        else:
            results["summary"] = "No relevant information found for this query."
            results["confidence"] = 0.0
        
        return results
    
    async def _plan_research(self, task: str) -> Dict[str, Any]:
        """Plan the research approach."""
        prompt = f"""Given this research task: "{task}"

Create a brief research plan including:
1. Key concepts to search for
2. Types of sources needed
3. Potential follow-up questions

Respond in JSON format:
{{
    "key_concepts": ["concept1", "concept2"],
    "source_types": ["type1", "type2"],
    "follow_ups": ["question1", "question2"]
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "key_concepts": [task],
                "source_types": ["general"],
                "follow_ups": []
            }
    
    async def _rag_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute a RAG query against the knowledge base."""
        if self.rag_pipeline is None:
            logger.warning("RAG pipeline not configured")
            return {"documents": [], "sources": []}
        
        try:
            results = await self.rag_pipeline.query(query, top_k=top_k)
            return {
                "documents": [doc.content for doc in results],
                "sources": [doc.metadata.get("source", "Unknown") for doc in results],
                "scores": [doc.score for doc in results]
            }
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {"documents": [], "sources": [], "error": str(e)}
    
    async def _web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute a web search."""
        # This is a placeholder - in production, integrate with a real search API
        logger.info(f"Web search for: {query}")
        
        # Simulate web search results
        return {
            "results": [
                {
                    "title": f"Search result for: {query}",
                    "snippet": "This is a placeholder for web search results.",
                    "url": "https://example.com"
                }
            ]
        }
    
    async def _summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize the given text."""
        prompt = f"""Summarize the following text in approximately {max_length} words:

{text}

Provide a clear, concise summary that captures the key points."""
        
        messages = [{"role": "user", "content": prompt}]
        return await self._call_llm(messages)
    
    async def _synthesize_findings(
        self,
        query: str,
        findings: List[str]
    ) -> Dict[str, Any]:
        """Synthesize multiple findings into a coherent response."""
        findings_text = "\n\n---\n\n".join(findings[:10])  # Limit to prevent context overflow
        
        prompt = f"""Research Query: {query}

Findings from various sources:
{findings_text}

Synthesize these findings into a comprehensive response that:
1. Directly answers the research query
2. Highlights key insights
3. Notes any contradictions or gaps
4. Provides a confidence level (0-1) based on source quality and agreement

Respond in JSON format:
{{
    "summary": "Your synthesized response",
    "key_insights": ["insight1", "insight2"],
    "gaps": ["gap1", "gap2"],
    "confidence": 0.85
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "summary": response,
                "key_insights": [],
                "gaps": [],
                "confidence": 0.5
            }
