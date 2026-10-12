# Contributing to Fraud Detection System

Thank you for your interest in contributing! This document provides guidelines for contributions.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black ruff mypy
   ```

## Code Style

- Use **Black** for formatting: `black src/ tests/`
- Use **Ruff** for linting: `ruff check src/ tests/`
- Use **mypy** for type checking: `mypy src/`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py -v
```

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with clear description

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

## Questions?

Open an issue for discussion.
