"""
Base Agent Abstract Class.

Defines the interface and common functionality for all agents
in the multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import time
import logging

from ..core.llm import BaseLLM, get_llm
from ..core.context import ConversationContext, AgentAction, MessageRole


logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    output: Any
    agent_name: str
    execution_time: float
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "agent_name": self.agent_name,
            "execution_time": self.execution_time,
            "actions_taken": self.actions_taken,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class Tool:
    """A tool that an agent can use."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function schema."""
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


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality for LLM interaction, tool use,
    and execution management.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: Optional[BaseLLM] = None,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        timeout: int = 60
    ):
        self.name = name
        self.description = description
        self.llm = llm or get_llm()
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_iterations = max_iterations
        self.timeout = timeout
        
        self.status = AgentStatus.IDLE
        self._actions_log: List[Dict[str, Any]] = []
    
    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return the default system prompt for this agent."""
        pass
    
    @abstractmethod
    async def _execute_task(
        self,
        task: str,
        context: ConversationContext,
        **kwargs
    ) -> Any:
        """Execute the agent's main task. Must be implemented by subclasses."""
        pass
    
    async def execute(
        self,
        task: str,
        context: Optional[ConversationContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute a task with this agent.
        
        Args:
            task: The task to execute
            context: Optional conversation context
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with execution details
        """
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        self._actions_log = []
        
        # Create context if not provided
        if context is None:
            context = ConversationContext()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task(task, context, **kwargs),
                timeout=self.timeout
            )
            
            self.status = AgentStatus.COMPLETED
            execution_time = time.time() - start_time
            
            # Record action in context
            context.add_agent_action(AgentAction(
                agent_name=self.name,
                action_type="execute",
                input_data=task,
                output_data=result,
                duration_ms=execution_time * 1000,
                success=True
            ))
            
            return AgentResult(
                success=True,
                output=result,
                agent_name=self.name,
                execution_time=execution_time,
                actions_taken=self._actions_log.copy()
            )
            
        except asyncio.TimeoutError:
            self.status = AgentStatus.TIMEOUT
            execution_time = time.time() - start_time
            error_msg = f"Agent {self.name} timed out after {self.timeout}s"
            
            logger.error(error_msg)
            
            return AgentResult(
                success=False,
                output=None,
                agent_name=self.name,
                execution_time=execution_time,
                actions_taken=self._actions_log.copy(),
                error=error_msg
            )
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            execution_time = time.time() - start_time
            error_msg = f"Agent {self.name} failed: {str(e)}"
            
            logger.exception(error_msg)
            
            context.add_agent_action(AgentAction(
                agent_name=self.name,
                action_type="execute",
                input_data=task,
                output_data=None,
                duration_ms=execution_time * 1000,
                success=False,
                error=str(e)
            ))
            
            return AgentResult(
                success=False,
                output=None,
                agent_name=self.name,
                execution_time=execution_time,
                actions_taken=self._actions_log.copy(),
                error=error_msg
            )
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        use_tools: bool = False,
        **kwargs
    ) -> str:
        """Call the LLM with the given messages."""
        # Add system prompt if not present
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        response = await self.llm.generate(messages, **kwargs)
        
        self._log_action("llm_call", {
            "messages_count": len(messages),
            "response_length": len(response.content)
        })
        
        return response.content
    
    async def _use_tool(self, tool_name: str, **params) -> Any:
        """Use a registered tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        start_time = time.time()
        
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**params)
            else:
                result = tool.function(**params)
            
            duration = time.time() - start_time
            
            self._log_action("tool_use", {
                "tool": tool_name,
                "params": params,
                "duration_ms": duration * 1000,
                "success": True
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            self._log_action("tool_use", {
                "tool": tool_name,
                "params": params,
                "duration_ms": duration * 1000,
                "success": False,
                "error": str(e)
            })
            
            raise
    
    def register_tool(self, tool: Tool) -> None:
        """Register a new tool for this agent."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool '{tool.name}' for agent '{self.name}'")
    
    def _log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """Log an action taken by this agent."""
        self._actions_log.append({
            "action_type": action_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible schemas for all tools."""
        return [tool.to_schema() for tool in self.tools.values()]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tools={list(self.tools.keys())})"
