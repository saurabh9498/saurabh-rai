"""
Orchestrator Agent.

The central coordinator that routes tasks to appropriate agents,
manages multi-agent workflows, and synthesizes results.
"""

from typing import List, Dict, Any, Optional, Type
import json
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseAgent, AgentResult, Tool
from .research_agent import ResearchAgent
from .analyst_agent import AnalystAgent
from .code_agent import CodeAgent
from ..core.context import ConversationContext, MessageRole
from ..core.llm import get_llm


logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TaskPlan:
    """A planned task for an agent."""
    agent_name: str
    task_description: str
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result from a multi-agent workflow."""
    success: bool
    final_output: Any
    agent_results: Dict[str, AgentResult]
    execution_order: List[str]
    total_time: float
    errors: List[str] = field(default_factory=list)


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent that coordinates multiple specialized agents.
    
    Responsibilities:
    - Task understanding and decomposition
    - Agent selection and routing
    - Workflow orchestration
    - Result synthesis
    - Quality assurance
    """
    
    def __init__(
        self,
        name: str = "orchestrator",
        agents: Optional[Dict[str, BaseAgent]] = None,
        enable_parallel: bool = False,
        **kwargs
    ):
        self.agents = agents or {}
        self.enable_parallel = enable_parallel
        
        super().__init__(
            name=name,
            description="Orchestrator agent for coordinating multi-agent workflows",
            **kwargs
        )
        
        # Register default agents if none provided
        if not self.agents:
            self._register_default_agents()
    
    def _default_system_prompt(self) -> str:
        available_agents = ", ".join(self.agents.keys()) if self.agents else "none"
        
        return f"""You are the Orchestrator Agent, responsible for coordinating complex tasks across multiple specialized agents.

Available agents: {available_agents}

Your responsibilities:
1. Understand user requests thoroughly
2. Decompose complex tasks into subtasks
3. Route subtasks to appropriate agents
4. Manage dependencies between tasks
5. Synthesize results into coherent responses
6. Ensure quality of final output

Agent capabilities:
- research_agent: Information retrieval, RAG queries, web search, summarization
- analyst_agent: Data analysis, trend analysis, comparisons, visualizations
- code_agent: Code generation, review, debugging, execution

When processing a request:
1. Analyze the request to understand requirements
2. Create a task plan with agent assignments
3. Execute tasks in appropriate order
4. Synthesize results
5. Provide a comprehensive response

Always aim for accuracy, completeness, and clarity in your responses."""
    
    def _register_default_agents(self) -> None:
        """Register the default specialized agents."""
        self.register_agent("research", ResearchAgent())
        self.register_agent("analyst", AnalystAgent())
        self.register_agent("code", CodeAgent())
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a new agent."""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")
    
    async def _execute_task(
        self,
        task: str,
        context: ConversationContext,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a complex task by orchestrating multiple agents.
        
        Args:
            task: The user's request
            context: Conversation context
            **kwargs: Additional arguments
            
        Returns:
            Synthesized results from all agents
        """
        # Step 1: Understand and plan
        task_plan = await self._plan_task(task, context)
        self._log_action("task_planning", {"plan": task_plan})
        
        if not task_plan["subtasks"]:
            # Simple task - handle directly
            return await self._handle_simple_task(task, context)
        
        # Step 2: Execute subtasks
        agent_results: Dict[str, AgentResult] = {}
        execution_order: List[str] = []
        
        if self.enable_parallel and not task_plan.get("has_dependencies", True):
            # Execute in parallel
            results = await self._execute_parallel(task_plan["subtasks"], context)
            agent_results.update(results)
            execution_order = list(results.keys())
        else:
            # Execute sequentially
            for subtask in task_plan["subtasks"]:
                agent_name = subtask["agent"]
                
                if agent_name not in self.agents:
                    logger.warning(f"Unknown agent: {agent_name}")
                    continue
                
                # Pass previous results as context
                subtask_context = {
                    "previous_results": agent_results,
                    **subtask.get("inputs", {})
                }
                
                result = await self.agents[agent_name].execute(
                    subtask["task"],
                    context,
                    **subtask_context
                )
                
                agent_results[agent_name] = result
                execution_order.append(agent_name)
        
        # Step 3: Synthesize results
        final_output = await self._synthesize_results(
            task,
            agent_results,
            context
        )
        
        return {
            "final_output": final_output,
            "agent_results": {k: v.to_dict() for k, v in agent_results.items()},
            "execution_order": execution_order,
            "task_plan": task_plan
        }
    
    async def _plan_task(
        self,
        task: str,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Plan how to execute the task using available agents."""
        agent_descriptions = "\n".join([
            f"- {name}: {agent.description}"
            for name, agent in self.agents.items()
        ])
        
        prompt = f"""Analyze this task and create an execution plan:

Task: {task}

Available Agents:
{agent_descriptions}

Create a plan that:
1. Identifies if multiple agents are needed
2. Breaks down complex tasks into subtasks
3. Assigns each subtask to the appropriate agent
4. Identifies dependencies between subtasks

Respond in JSON format:
{{
    "understanding": "Your understanding of the task",
    "complexity": "simple|moderate|complex",
    "subtasks": [
        {{
            "agent": "agent_name",
            "task": "Subtask description",
            "priority": "high|medium|low",
            "depends_on": []
        }}
    ],
    "has_dependencies": true|false,
    "expected_output": "Description of expected final output"
}}

If this is a simple task that doesn't need multiple agents, return an empty subtasks array."""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "understanding": task,
                "complexity": "simple",
                "subtasks": [],
                "has_dependencies": False,
                "expected_output": "Direct response"
            }
    
    async def _handle_simple_task(
        self,
        task: str,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Handle a simple task that doesn't need agent coordination."""
        messages = context.get_messages_for_llm()
        messages.append({"role": "user", "content": task})
        
        response = await self._call_llm(messages)
        
        return {
            "final_output": response,
            "agent_results": {},
            "execution_order": [],
            "task_plan": {"complexity": "simple", "subtasks": []}
        }
    
    async def _execute_parallel(
        self,
        subtasks: List[Dict[str, Any]],
        context: ConversationContext
    ) -> Dict[str, AgentResult]:
        """Execute multiple subtasks in parallel."""
        async def execute_subtask(subtask: Dict[str, Any]) -> tuple:
            agent_name = subtask["agent"]
            if agent_name not in self.agents:
                return agent_name, AgentResult(
                    success=False,
                    output=None,
                    agent_name=agent_name,
                    execution_time=0,
                    error=f"Unknown agent: {agent_name}"
                )
            
            result = await self.agents[agent_name].execute(
                subtask["task"],
                context,
                **subtask.get("inputs", {})
            )
            return agent_name, result
        
        # Execute all subtasks concurrently
        tasks = [execute_subtask(subtask) for subtask in subtasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        agent_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel execution error: {result}")
                continue
            agent_name, agent_result = result
            agent_results[agent_name] = agent_result
        
        return agent_results
    
    async def _synthesize_results(
        self,
        original_task: str,
        agent_results: Dict[str, AgentResult],
        context: ConversationContext
    ) -> str:
        """Synthesize results from multiple agents into a coherent response."""
        if not agent_results:
            return "No results to synthesize."
        
        # Format agent results for synthesis
        results_summary = []
        for agent_name, result in agent_results.items():
            if result.success:
                results_summary.append(f"""
Agent: {agent_name}
Status: Success
Output: {json.dumps(result.output, indent=2) if isinstance(result.output, dict) else result.output}
""")
            else:
                results_summary.append(f"""
Agent: {agent_name}
Status: Failed
Error: {result.error}
""")
        
        results_text = "\n---\n".join(results_summary)
        
        prompt = f"""Synthesize these agent results into a comprehensive response:

Original Task: {original_task}

Agent Results:
{results_text}

Create a unified response that:
1. Directly addresses the original task
2. Integrates insights from all successful agents
3. Notes any limitations or failed components
4. Is well-structured and easy to understand

Provide the synthesized response:"""
        
        messages = [{"role": "user", "content": prompt}]
        return await self._call_llm(messages)
    
    async def execute_workflow(
        self,
        task: str,
        context: Optional[ConversationContext] = None,
        agents: Optional[List[str]] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute a complete workflow with specified agents.
        
        Args:
            task: The main task
            context: Optional conversation context
            agents: Optional list of specific agents to use
            
        Returns:
            WorkflowResult with all execution details
        """
        import time
        start_time = time.time()
        
        context = context or ConversationContext()
        
        # Add user message to context
        context.add_message(MessageRole.USER, task)
        
        result = await self.execute(task, context, **kwargs)
        
        total_time = time.time() - start_time
        
        # Add assistant response to context
        if result.success:
            context.add_message(
                MessageRole.ASSISTANT,
                result.output.get("final_output", str(result.output))
            )
        
        return WorkflowResult(
            success=result.success,
            final_output=result.output.get("final_output") if result.success else None,
            agent_results=result.output.get("agent_results", {}) if result.success else {},
            execution_order=result.output.get("execution_order", []) if result.success else [],
            total_time=total_time,
            errors=[result.error] if result.error else []
        )
    
    def get_available_agents(self) -> Dict[str, str]:
        """Get list of available agents and their descriptions."""
        return {
            name: agent.description
            for name, agent in self.agents.items()
        }
