"""
Tool Registry.

Centralized registry for managing tools available to agents.
Provides registration, discovery, and execution of tools.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import inspect


logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    COMMUNICATION = "communication"
    UTILITY = "utility"
    EXTERNAL = "external"


@dataclass
class ToolDefinition:
    """Definition of a tool."""
    name: str
    description: str
    function: Callable
    category: ToolCategory = ToolCategory.UTILITY
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    returns: str = "Any"
    examples: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    rate_limit: Optional[int] = None  # calls per minute
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params
                }
            }
        }
    
    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Provides:
    - Tool registration and discovery
    - Schema generation for LLM function calling
    - Tool execution with error handling
    - Rate limiting and access control
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, ToolDefinition] = {}
            cls._instance._call_counts: Dict[str, List[float]] = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        category: ToolCategory = ToolCategory.UTILITY,
        parameters: Optional[Dict[str, Any]] = None,
        required_params: Optional[List[str]] = None,
        **kwargs
    ) -> ToolDefinition:
        """
        Register a new tool.
        
        Args:
            name: Unique tool name
            description: Tool description for LLM
            function: The callable to execute
            category: Tool category
            parameters: Parameter definitions
            required_params: Required parameters
            
        Returns:
            The registered ToolDefinition
        """
        # Auto-generate parameters from function signature if not provided
        if parameters is None:
            parameters = self._extract_parameters(function)
        
        if required_params is None:
            required_params = self._extract_required_params(function)
        
        tool = ToolDefinition(
            name=name,
            description=description,
            function=function,
            category=category,
            parameters=parameters,
            required_params=required_params,
            **kwargs
        )
        
        self._tools[name] = tool
        logger.info(f"Registered tool: {name} ({category.value})")
        
        return tool
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True
    ) -> List[ToolDefinition]:
        """List all registered tools."""
        tools = list(self._tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        
        return tools
    
    def get_schemas(
        self,
        format: str = "openai",
        tools: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tool schemas for LLM function calling.
        
        Args:
            format: Schema format (openai, anthropic)
            tools: Optional list of specific tools
            
        Returns:
            List of tool schemas
        """
        if tools:
            tool_defs = [self._tools[t] for t in tools if t in self._tools]
        else:
            tool_defs = [t for t in self._tools.values() if t.enabled]
        
        if format == "openai":
            return [t.to_openai_schema() for t in tool_defs]
        elif format == "anthropic":
            return [t.to_anthropic_schema() for t in tool_defs]
        else:
            raise ValueError(f"Unknown schema format: {format}")
    
    async def execute(
        self,
        name: str,
        **kwargs
    ) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        
        if not tool.enabled:
            raise ValueError(f"Tool is disabled: {name}")
        
        # Check rate limit
        if tool.rate_limit:
            if not self._check_rate_limit(name, tool.rate_limit):
                raise RuntimeError(f"Rate limit exceeded for tool: {name}")
        
        # Execute tool
        try:
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.function(**kwargs))
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {name} - {e}")
            raise
    
    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter definitions from function signature."""
        sig = inspect.signature(func)
        parameters = {}
        
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        for name, param in sig.parameters.items():
            if name in ["self", "cls"]:
                continue
            
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                param_type = type_mapping.get(param.annotation, "string")
            
            param_def = {"type": param_type}
            
            if param.default != inspect.Parameter.empty:
                param_def["default"] = param.default
            
            parameters[name] = param_def
        
        return parameters
    
    def _extract_required_params(self, func: Callable) -> List[str]:
        """Extract required parameters from function signature."""
        sig = inspect.signature(func)
        required = []
        
        for name, param in sig.parameters.items():
            if name in ["self", "cls"]:
                continue
            if param.default == inspect.Parameter.empty:
                required.append(name)
        
        return required
    
    def _check_rate_limit(self, name: str, limit: int) -> bool:
        """Check if tool is within rate limit."""
        import time
        
        now = time.time()
        window = 60  # 1 minute window
        
        if name not in self._call_counts:
            self._call_counts[name] = []
        
        # Remove old calls
        self._call_counts[name] = [
            t for t in self._call_counts[name]
            if now - t < window
        ]
        
        # Check limit
        if len(self._call_counts[name]) >= limit:
            return False
        
        # Record call
        self._call_counts[name].append(now)
        return True


# Global registry instance
registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.UTILITY,
    **kwargs
):
    """
    Decorator to register a function as a tool.
    
    Usage:
        @tool(name="my_tool", description="Does something useful")
        def my_function(param1: str, param2: int = 10):
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        
        registry.register(
            name=tool_name,
            description=tool_desc,
            function=func,
            category=category,
            **kwargs
        )
        
        return func
    
    return decorator
