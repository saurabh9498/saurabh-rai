"""
Unit tests for agents module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.agents.base_agent import BaseAgent, Tool, AgentResult, AgentStatus
from src.agents.research_agent import ResearchAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.code_agent import CodeAgent
from src.agents.orchestrator import OrchestratorAgent
from src.core.context import ConversationContext


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_tool_registration(self):
        """Test that tools can be registered to agents."""
        class ConcreteAgent(BaseAgent):
            def _default_system_prompt(self):
                return "Test prompt"
            
            async def _execute_task(self, task, context, **kwargs):
                return {"result": "test"}
        
        agent = ConcreteAgent(name="test_agent", description="Test")
        
        tool = Tool(
            name="test_tool",
            description="A test tool",
            function=lambda x: x,
            parameters={"x": {"type": "string"}}
        )
        
        agent.register_tool(tool)
        assert "test_tool" in agent.tools
        assert agent.tools["test_tool"].description == "A test tool"
    
    def test_tool_schema_generation(self):
        """Test OpenAI-compatible schema generation."""
        class ConcreteAgent(BaseAgent):
            def _default_system_prompt(self):
                return "Test prompt"
            
            async def _execute_task(self, task, context, **kwargs):
                return {}
        
        agent = ConcreteAgent(name="test", description="Test")
        
        tool = Tool(
            name="search",
            description="Search for information",
            function=lambda q: q,
            parameters={"query": {"type": "string", "description": "Search query"}},
            required_params=["query"]
        )
        
        agent.register_tool(tool)
        schemas = agent.get_tool_schemas()
        
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"
        assert "query" in schemas[0]["function"]["parameters"]["properties"]


class TestResearchAgent:
    """Tests for ResearchAgent."""
    
    @pytest.fixture
    def research_agent(self):
        """Create a research agent with mocked LLM."""
        with patch('src.agents.research_agent.get_llm') as mock_llm:
            mock_llm.return_value = AsyncMock()
            agent = ResearchAgent(name="test_research")
            return agent
    
    def test_initialization(self, research_agent):
        """Test research agent initializes with correct tools."""
        assert "rag_query" in research_agent.tools
        assert "summarize" in research_agent.tools
        assert research_agent.name == "test_research"
    
    def test_web_search_tool_registration(self):
        """Test web search tool is registered when enabled."""
        with patch('src.agents.research_agent.get_llm'):
            agent = ResearchAgent(web_search_enabled=True)
            assert "web_search" in agent.tools
            
            agent_no_web = ResearchAgent(web_search_enabled=False)
            assert "web_search" not in agent_no_web.tools
    
    @pytest.mark.asyncio
    async def test_execute_returns_result(self, research_agent):
        """Test that execute returns an AgentResult."""
        research_agent.llm.generate = AsyncMock(return_value=Mock(content='{"key_concepts": ["test"]}'))
        
        context = ConversationContext()
        result = await research_agent.execute("Test query", context)
        
        assert isinstance(result, AgentResult)
        assert result.agent_name == "test_research"


class TestAnalystAgent:
    """Tests for AnalystAgent."""
    
    @pytest.fixture
    def analyst_agent(self):
        """Create an analyst agent with mocked LLM."""
        with patch('src.agents.analyst_agent.get_llm') as mock_llm:
            mock_llm.return_value = AsyncMock()
            agent = AnalystAgent(name="test_analyst")
            return agent
    
    def test_initialization(self, analyst_agent):
        """Test analyst agent initializes with correct tools."""
        assert "query_data" in analyst_agent.tools
        assert "statistical_analysis" in analyst_agent.tools
        assert "trend_analysis" in analyst_agent.tools
        assert "compare" in analyst_agent.tools
    
    def test_default_system_prompt(self, analyst_agent):
        """Test system prompt contains key capabilities."""
        prompt = analyst_agent._default_system_prompt()
        assert "Data Analysis" in prompt
        assert "Trend" in prompt


class TestCodeAgent:
    """Tests for CodeAgent."""
    
    @pytest.fixture
    def code_agent(self):
        """Create a code agent with mocked LLM."""
        with patch('src.agents.code_agent.get_llm') as mock_llm:
            mock_llm.return_value = AsyncMock()
            agent = CodeAgent(name="test_code", enable_execution=True)
            return agent
    
    def test_initialization(self, code_agent):
        """Test code agent initializes with correct tools."""
        assert "generate_code" in code_agent.tools
        assert "review_code" in code_agent.tools
        assert "debug_code" in code_agent.tools
        assert "execute_code" in code_agent.tools
    
    def test_execution_disabled(self):
        """Test execute_code tool is not registered when disabled."""
        with patch('src.agents.code_agent.get_llm'):
            agent = CodeAgent(enable_execution=False)
            assert "execute_code" not in agent.tools
    
    def test_supported_languages(self, code_agent):
        """Test supported languages list."""
        assert "python" in code_agent.SUPPORTED_LANGUAGES
        assert "javascript" in code_agent.SUPPORTED_LANGUAGES


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator with mocked agents."""
        with patch('src.agents.orchestrator.get_llm') as mock_llm:
            mock_llm.return_value = AsyncMock()
            
            # Create with empty agents dict to avoid loading real agents
            agent = OrchestratorAgent(name="test_orchestrator", agents={})
            return agent
    
    def test_agent_registration(self, orchestrator):
        """Test agents can be registered."""
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.description = "Test agent"
        
        orchestrator.register_agent("test", mock_agent)
        
        assert "test" in orchestrator.agents
        assert orchestrator.agents["test"] == mock_agent
    
    def test_get_available_agents(self, orchestrator):
        """Test getting list of available agents."""
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.description = "Test description"
        
        orchestrator.register_agent("test", mock_agent)
        
        agents = orchestrator.get_available_agents()
        assert "test" in agents
        assert agents["test"] == "Test description"
    
    @pytest.mark.asyncio
    async def test_simple_task_handling(self, orchestrator):
        """Test handling of simple tasks without agent coordination."""
        orchestrator.llm.generate = AsyncMock(
            return_value=Mock(content='{"complexity": "simple", "subtasks": []}')
        )
        
        context = ConversationContext()
        result = await orchestrator.execute("Simple question", context)
        
        assert result.success


class TestConversationContext:
    """Tests for ConversationContext."""
    
    def test_message_addition(self):
        """Test adding messages to context."""
        from src.core.context import MessageRole
        
        context = ConversationContext()
        msg = context.add_message(MessageRole.USER, "Hello")
        
        assert len(context.messages) == 1
        assert context.messages[0].content == "Hello"
        assert context.messages[0].role == MessageRole.USER
    
    def test_context_trimming(self):
        """Test that context is trimmed when max_messages exceeded."""
        from src.core.context import MessageRole
        
        context = ConversationContext(max_messages=5)
        
        for i in range(10):
            context.add_message(MessageRole.USER, f"Message {i}")
        
        assert len(context.messages) <= 5
    
    def test_get_messages_for_llm(self):
        """Test formatting messages for LLM input."""
        from src.core.context import MessageRole
        
        context = ConversationContext()
        context.add_message(MessageRole.SYSTEM, "You are helpful")
        context.add_message(MessageRole.USER, "Hello")
        context.add_message(MessageRole.ASSISTANT, "Hi there!")
        
        messages = context.get_messages_for_llm()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_variable_storage(self):
        """Test context variable storage."""
        context = ConversationContext()
        
        context.set_variable("key", "value")
        assert context.get_variable("key") == "value"
        assert context.get_variable("missing", "default") == "default"
    
    def test_serialization(self):
        """Test context serialization and deserialization."""
        from src.core.context import MessageRole
        
        context = ConversationContext()
        context.add_message(MessageRole.USER, "Test message")
        context.set_variable("test_key", "test_value")
        
        # Serialize
        data = context.to_dict()
        
        # Deserialize
        restored = ConversationContext.from_dict(data)
        
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Test message"
        assert restored.get_variable("test_key") == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
