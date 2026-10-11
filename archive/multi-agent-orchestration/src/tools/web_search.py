"""
Web Search Tool.

Provides web search capabilities using various search APIs.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import asyncio
import aiohttp
from abc import ABC, abstractmethod

from .registry import tool, ToolCategory


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str = ""
    published_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date
        }


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """Execute a search query."""
        pass


class TavilySearchProvider(SearchProvider):
    """Tavily AI search provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"
    
    async def search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """Search using Tavily API."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": num_results
            }
            
            async with session.post(self.base_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Tavily search failed: {response.status}")
                    return []
                
                data = await response.json()
                
                results = []
                for item in data.get("results", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                        source="tavily"
                    ))
                
                return results


class SerperSearchProvider(SearchProvider):
    """Serper.dev Google search provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
    
    async def search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """Search using Serper API."""
        async with aiohttp.ClientSession() as session:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": num_results
            }
            
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    logger.error(f"Serper search failed: {response.status}")
                    return []
                
                data = await response.json()
                
                results = []
                for item in data.get("organic", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="serper"
                    ))
                
                return results


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search provider (no API key required)."""
    
    async def search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import AsyncDDGS
            
            results = []
            async with AsyncDDGS() as ddgs:
                async for r in ddgs.text(query, max_results=num_results):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="duckduckgo"
                    ))
            
            return results
            
        except ImportError:
            logger.warning("duckduckgo_search not installed")
            return []


class WebSearchTool:
    """
    Web search tool with multiple provider support.
    
    Provides a unified interface for web search across different
    search providers with fallback capabilities.
    """
    
    def __init__(
        self,
        providers: Optional[List[SearchProvider]] = None,
        default_num_results: int = 5
    ):
        self.providers = providers or [DuckDuckGoSearchProvider()]
        self.default_num_results = default_num_results
    
    async def search(
        self,
        query: str,
        num_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a web search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Search results
        """
        num_results = num_results or self.default_num_results
        
        # Try each provider until one succeeds
        for provider in self.providers:
            try:
                results = await provider.search(query, num_results)
                if results:
                    return {
                        "success": True,
                        "query": query,
                        "results": [r.to_dict() for r in results],
                        "provider": provider.__class__.__name__
                    }
            except Exception as e:
                logger.warning(f"Search provider failed: {e}")
                continue
        
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": "All search providers failed"
        }
    
    async def search_and_summarize(
        self,
        query: str,
        llm,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search and summarize results using LLM.
        
        Args:
            query: Search query
            llm: LLM instance for summarization
            num_results: Number of results
            
        Returns:
            Search results with summary
        """
        search_results = await self.search(query, num_results)
        
        if not search_results["success"] or not search_results["results"]:
            return search_results
        
        # Format results for summarization
        results_text = "\n\n".join([
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['snippet']}"
            for r in search_results["results"]
        ])
        
        prompt = f"""Based on these search results for "{query}":

{results_text}

Provide a comprehensive summary that answers the query. Include relevant facts and cite sources when possible."""
        
        messages = [{"role": "user", "content": prompt}]
        response = await llm.generate(messages)
        
        search_results["summary"] = response.content
        return search_results


# Register as tool
@tool(
    name="web_search",
    description="Search the web for current information. Use this for recent events, facts, or when you need up-to-date information.",
    category=ToolCategory.SEARCH,
    parameters={
        "query": {
            "type": "string",
            "description": "The search query"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return",
            "default": 5
        }
    },
    required_params=["query"]
)
async def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Execute a web search."""
    tool = WebSearchTool()
    return await tool.search(query, num_results)
