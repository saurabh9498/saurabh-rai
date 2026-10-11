"""
LLM Provider Abstraction Layer.

This module provides a unified interface for different LLM providers,
enabling seamless switching between OpenAI, Anthropic, and local models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from .config import settings, LLMConfig


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM."""
        pass
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convert dict messages to LangChain message objects."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        
        return converted


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = ChatOpenAI(
            model=model,
            api_key=api_key or settings.llm.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI."""
        langchain_messages = self._convert_messages(messages)
        
        # Update parameters if provided
        if temperature is not None:
            self.client.temperature = temperature
        if max_tokens is not None:
            self.client.max_tokens = max_tokens
        
        response = await self.client.ainvoke(langchain_messages, **kwargs)
        
        return LLMResponse(
            content=response.content,
            model=self.model,
            usage=response.response_metadata.get("usage", {}),
            finish_reason=response.response_metadata.get("finish_reason"),
            raw_response=response
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response using OpenAI."""
        langchain_messages = self._convert_messages(messages)
        
        if temperature is not None:
            self.client.temperature = temperature
        if max_tokens is not None:
            self.client.max_tokens = max_tokens
        
        async for chunk in self.client.astream(langchain_messages, **kwargs):
            if chunk.content:
                yield chunk.content


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation."""
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = ChatAnthropic(
            model=model,
            api_key=api_key or settings.llm.anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Anthropic Claude."""
        langchain_messages = self._convert_messages(messages)
        
        if temperature is not None:
            self.client.temperature = temperature
        if max_tokens is not None:
            self.client.max_tokens = max_tokens
        
        response = await self.client.ainvoke(langchain_messages, **kwargs)
        
        return LLMResponse(
            content=response.content,
            model=self.model,
            usage=response.response_metadata.get("usage", {}),
            finish_reason=response.response_metadata.get("stop_reason"),
            raw_response=response
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response using Anthropic Claude."""
        langchain_messages = self._convert_messages(messages)
        
        if temperature is not None:
            self.client.temperature = temperature
        if max_tokens is not None:
            self.client.max_tokens = max_tokens
        
        async for chunk in self.client.astream(langchain_messages, **kwargs):
            if chunk.content:
                yield chunk.content


class LLMFactory:
    """Factory for creating LLM instances."""
    
    _providers = {
        LLMProvider.OPENAI: OpenAILLM,
        LLMProvider.ANTHROPIC: AnthropicLLM,
    }
    
    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Create an LLM instance based on configuration."""
        provider = provider or settings.llm.provider
        model = model or settings.llm.model_name
        
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        llm_class = cls._providers.get(provider_enum)
        if not llm_class:
            raise ValueError(f"No implementation for provider: {provider}")
        
        return llm_class(
            model=model,
            temperature=kwargs.get("temperature", settings.llm.temperature),
            max_tokens=kwargs.get("max_tokens", settings.llm.max_tokens),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )


# Convenience function
def get_llm(provider: Optional[str] = None, **kwargs) -> BaseLLM:
    """Get an LLM instance."""
    return LLMFactory.create(provider=provider, **kwargs)
