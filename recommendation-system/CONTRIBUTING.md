# Contributing to Real-Time Personalization Engine

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- NVIDIA GPU (optional, for GPU-accelerated development)
- Git

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/recommendation-engine.git
cd recommendation-engine
git remote add upstream https://github.com/ORIGINAL_OWNER/recommendation-engine.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your local settings
```

### 4. Start Development Services

```bash
docker-compose -f docker-compose.dev.yml up -d
```

### 5. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-user-embeddings` - New features
- `fix/retrieval-timeout` - Bug fixes
- `docs/api-reference` - Documentation
- `refactor/feature-store` - Code refactoring
- `test/dlrm-unit-tests` - Test additions

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(retrieval): add HNSW index support for faster ANN search

fix(api): handle timeout errors gracefully in batch endpoint

docs(readme): add performance benchmarks section
```

## Pull Request Process

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Run Quality Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run tests
pytest --cov=src
```

### 3. Create Pull Request

- Use a descriptive title following commit message conventions
- Fill out the PR template completely
- Link related issues
- Request reviews from maintainers

### 4. PR Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] No breaking changes (or they are documented)
- [ ] Performance impact is considered

## Coding Standards

### Python Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with these tools:

- **Black** for formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Code Organization

```
src/
├── models/          # ML model implementations
├── features/        # Feature engineering
├── serving/         # API and serving logic
├── data/            # Data loading utilities
└── utils/           # Shared utilities
```

### Documentation

- All public functions need docstrings (Google style)
- Complex algorithms should have inline comments
- Update README for user-facing changes

Example docstring:
```python
def get_recommendations(
    user_id: str,
    num_items: int = 10,
    context: Optional[Dict[str, Any]] = None,
) -> List[RecommendedItem]:
    """Get personalized recommendations for a user.

    Args:
        user_id: Unique identifier for the user.
        num_items: Number of recommendations to return.
        context: Optional context (device, page, etc.).

    Returns:
        List of recommended items with scores.

    Raises:
        UserNotFoundError: If user_id doesn't exist.
        ServiceUnavailableError: If backend services are down.

    Example:
        >>> recs = get_recommendations("user_123", num_items=5)
        >>> print(recs[0].item_id)
        'item_456'
    """
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (require services)
└── load/              # Load/performance tests
```

### Writing Tests

```python
import pytest
from src.models.dlrm import DLRM, DLRMConfig


class TestDLRM:
    """Tests for DLRM model."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = DLRMConfig(num_sparse_features=5)
        return DLRM(config)

    def test_forward_pass(self, model):
        """Test model forward pass produces correct output shape."""
        sparse = torch.randint(0, 100, (8, 5))
        dense = torch.randn(8, 13)

        output = model(sparse, dense)

        assert output.shape == (8, 1)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_batch_sizes(self, model, batch_size):
        """Test model handles various batch sizes."""
        # ...
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_dlrm.py

# With coverage
pytest --cov=src --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu
```

### Test Coverage

- Maintain minimum 80% code coverage
- Critical paths (serving, model inference) should have 90%+ coverage
- Add tests for all bug fixes

## Performance Considerations

When contributing performance-sensitive code:

1. **Benchmark** changes using `scripts/benchmark.py`
2. **Profile** with `py-spy` or `cProfile`
3. **Document** performance characteristics
4. **Test** with production-like data volumes

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Security**: Email security@example.com (do not open public issues)

## Recognition

Contributors are recognized in:
- CHANGELOG.md for each release
- GitHub contributors page
- Annual contributor highlights

Thank you for contributing! 🎉
