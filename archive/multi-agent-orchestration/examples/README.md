# Examples

This directory contains example scripts demonstrating how to use the multi-agent system.

## Available Examples

### 1. Basic Completion (`basic_completion.py`)

Demonstrates fundamental LLM interactions:
- Simple prompt completion
- Token tracking and cost estimation
- Response caching
- Rate limiting

```bash
python examples/basic_completion.py
```

### 2. Chat Session (`chat_session.py`)

Shows multi-turn conversation handling:
- Context management across turns
- Session handling
- Multiple concurrent sessions
- Interactive chat mode

```bash
python examples/chat_session.py
```

### 3. Chain Prompts (`chain_prompts.py`)

Demonstrates prompt chaining workflows:
- Sequential prompt execution
- Variable passing between steps
- Research and summarization pipeline
- Content creation workflow
- Code review chain
- Parallel execution

```bash
python examples/chain_prompts.py
```

## Prerequisites

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Concepts Demonstrated

### Rate Limiting
```python
from src.utils import rate_limiter

# Acquire permission before API call
await rate_limiter.acquire("openai", tokens=1000)
```

### Token Tracking
```python
from src.utils import token_tracker

# Count tokens before request
tokens = token_tracker.count_tokens(messages=messages, model="gpt-4o")

# Get usage stats
stats = token_tracker.get_stats()
print(f"Total cost: ${stats.total_cost:.4f}")
```

### Caching
```python
from src.utils import llm_cache

# Check cache before API call
cached = llm_cache.get(model="gpt-4o", messages=messages)
if cached:
    return cached  # Skip API call
```

### Logging
```python
from src.utils import get_logger, set_request_context

logger = get_logger(__name__)
set_request_context(request_id="abc123", agent_name="research")

logger.info("Processing request")
```

## Running All Examples

```bash
# Run all examples in sequence
for example in basic_completion chat_session chain_prompts; do
    echo "Running $example..."
    python examples/${example}.py
done
```

## Output

Each example prints:
- Operation results
- Token usage statistics
- Cost estimates
- Cache hit rates

## Extending Examples

To create new examples:

1. Import required utilities:
```python
from src.utils import (
    setup_logging,
    get_logger,
    token_tracker,
    rate_limiter,
    llm_cache,
)
```

2. Set up logging:
```python
setup_logging(level="INFO")
logger = get_logger(__name__)
```

3. Use async/await for LLM calls:
```python
async def main():
    client = LLMClient(provider="openai", model="gpt-4o-mini")
    response = await client.complete(prompt="Hello!")
    print(response)

asyncio.run(main())
```
