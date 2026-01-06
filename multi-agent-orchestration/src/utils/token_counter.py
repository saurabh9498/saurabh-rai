"""
Token Counter Utility

Provides accurate token counting for various LLM providers.
Tracks usage and estimates costs across conversations.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Pricing per 1K tokens (as of 2024)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
}


@dataclass
class TokenUsage:
    """Represents token usage for a single request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def cost(self) -> float:
        """Calculate cost based on model pricing."""
        pricing = MODEL_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = (self.prompt_tokens / 1000) * pricing["input"]
        output_cost = (self.completion_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class UsageStats:
    """Aggregated usage statistics."""
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add(self, usage: TokenUsage) -> None:
        """Add a usage record to stats."""
        self.total_requests += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.cost
        
        if usage.model not in self.by_model:
            self.by_model[usage.model] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
        
        self.by_model[usage.model]["requests"] += 1
        self.by_model[usage.model]["prompt_tokens"] += usage.prompt_tokens
        self.by_model[usage.model]["completion_tokens"] += usage.completion_tokens
        self.by_model[usage.model]["total_tokens"] += usage.total_tokens
        self.by_model[usage.model]["cost"] += usage.cost


class TokenCounter(ABC):
    """Abstract base class for token counters."""
    
    @abstractmethod
    def count(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages."""
        pass


class TiktokenCounter(TokenCounter):
    """
    Token counter using OpenAI's tiktoken library.
    
    Provides accurate counting for OpenAI models.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._encoding = None
    
    @property
    def encoding(self):
        """Lazy load tiktoken encoding."""
        if self._encoding is None:
            try:
                import tiktoken
                try:
                    self._encoding = tiktoken.encoding_for_model(self.model)
                except KeyError:
                    self._encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not installed, using approximate counting")
                return None
        return self._encoding
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding is None:
            # Fallback: approximate 4 chars per token
            return len(text) // 4
        return len(self.encoding.encode(text))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in chat messages.
        
        Accounts for message formatting overhead.
        """
        if self.encoding is None:
            return sum(len(str(m)) // 4 for m in messages)
        
        # Token overhead per message (varies by model)
        tokens_per_message = 4  # <|start|>role<|end|>content
        tokens_per_name = -1  # If name is present
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        
        num_tokens += 3  # <|start|>assistant<|message|>
        return num_tokens


class ApproximateCounter(TokenCounter):
    """
    Approximate token counter for when tiktoken is not available.
    
    Uses heuristics based on typical tokenization patterns.
    """
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
    
    def count(self, text: str) -> int:
        """Approximate token count."""
        # More sophisticated approximation
        # Count words and special characters
        words = len(re.findall(r'\b\w+\b', text))
        special_chars = len(re.findall(r'[^\w\s]', text))
        whitespace_runs = len(re.findall(r'\s+', text))
        
        # Heuristic: words + some special chars + formatting
        estimated = words + (special_chars * 0.5) + (whitespace_runs * 0.1)
        
        # Fallback to char-based if text is very different
        char_estimate = len(text) / self.chars_per_token
        
        return int(max(estimated, char_estimate))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Approximate token count for messages."""
        total = 0
        for message in messages:
            # Add overhead per message
            total += 4
            for key, value in message.items():
                total += self.count(str(value))
        total += 3  # Assistant priming
        return total


class TokenTracker:
    """
    Tracks token usage across requests and sessions.
    
    Provides cost estimation, usage analytics, and budget alerts.
    """
    
    def __init__(
        self,
        budget_limit: Optional[float] = None,
        token_limit: Optional[int] = None,
    ):
        self.budget_limit = budget_limit
        self.token_limit = token_limit
        self.usage_history: List[TokenUsage] = []
        self._counters: Dict[str, TokenCounter] = {}
        self._stats = UsageStats()
    
    def get_counter(self, model: str) -> TokenCounter:
        """Get appropriate counter for model."""
        if model not in self._counters:
            try:
                self._counters[model] = TiktokenCounter(model)
            except Exception:
                self._counters[model] = ApproximateCounter()
        return self._counters[model]
    
    def count_tokens(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o",
    ) -> int:
        """Count tokens in text or messages."""
        counter = self.get_counter(model)
        
        if messages is not None:
            return counter.count_messages(messages)
        elif text is not None:
            return counter.count(text)
        return 0
    
    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> TokenUsage:
        """Record token usage from an API response."""
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
        )
        
        self.usage_history.append(usage)
        self._stats.add(usage)
        
        # Check limits
        self._check_limits(usage)
        
        logger.debug(
            f"Recorded usage: {usage.total_tokens} tokens "
            f"(${usage.cost:.4f}) for {model}"
        )
        
        return usage
    
    def _check_limits(self, usage: TokenUsage) -> None:
        """Check if usage limits are exceeded."""
        if self.budget_limit and self._stats.total_cost >= self.budget_limit:
            logger.warning(
                f"Budget limit reached: ${self._stats.total_cost:.2f} "
                f">= ${self.budget_limit:.2f}"
            )
        
        if self.token_limit and self._stats.total_tokens >= self.token_limit:
            logger.warning(
                f"Token limit reached: {self._stats.total_tokens:,} "
                f">= {self.token_limit:,}"
            )
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        max_completion_tokens: int,
        model: str,
    ) -> float:
        """Estimate cost before making a request."""
        pricing = MODEL_PRICING.get(model, {"input": 0.01, "output": 0.03})
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (max_completion_tokens / 1000) * pricing["output"]
        return input_cost + output_cost
    
    def get_stats(
        self,
        since: Optional[datetime] = None,
        model: Optional[str] = None,
    ) -> UsageStats:
        """Get usage statistics, optionally filtered."""
        if since is None and model is None:
            return self._stats
        
        # Filter and recalculate
        stats = UsageStats()
        for usage in self.usage_history:
            if since and usage.timestamp < since:
                continue
            if model and usage.model != model:
                continue
            stats.add(usage)
        
        return stats
    
    def get_daily_stats(self) -> UsageStats:
        """Get stats for the current day."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_stats(since=today)
    
    def get_monthly_stats(self) -> UsageStats:
        """Get stats for the current month."""
        first_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self.get_stats(since=first_of_month)
    
    def export_usage(self, filepath: str) -> None:
        """Export usage history to JSON file."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "summary": {
                "total_requests": self._stats.total_requests,
                "total_tokens": self._stats.total_tokens,
                "total_cost": self._stats.total_cost,
                "by_model": self._stats.by_model,
            },
            "history": [
                {
                    "timestamp": u.timestamp.isoformat(),
                    "model": u.model,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                    "cost": u.cost,
                }
                for u in self.usage_history
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported usage to {filepath}")
    
    def reset(self) -> None:
        """Reset all tracked usage."""
        self.usage_history.clear()
        self._stats = UsageStats()
        logger.info("Usage tracker reset")


# Global token tracker instance
token_tracker = TokenTracker()
