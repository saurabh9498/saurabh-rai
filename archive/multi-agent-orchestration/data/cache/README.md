# Cache Directory

This directory stores cached LLM responses to improve performance and reduce API costs.

## Contents

- Response cache files (`.pkl`)
- Cache index (`.json`)

## Usage

The cache is automatically managed by `src/utils/cache.py`. Files are created when:
- LLM responses are cached (with configurable TTL)
- Embeddings are cached for reuse

## Cleanup

To clear the cache:
```bash
make clean-cache
# or
rm -rf data/cache/*
```

## Configuration

Cache settings can be configured in `config/logging_config.yaml` or programmatically:

```python
from src.utils import LLMCache, DiskCache

cache = LLMCache(
    backend=DiskCache(cache_dir="./data/cache"),
    default_ttl=3600,  # 1 hour
)
```

## Note

This directory is git-ignored except for this README. Cache files should not be committed.
