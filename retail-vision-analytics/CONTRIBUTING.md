# Contributing to Retail Vision Analytics

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.0+
- Docker (for containerized development)
- Git

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/retail-vision-analytics.git
   cd retail-vision-analytics
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Copy environment template:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Code Standards

### Style Guide

We follow PEP 8 with these tools:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Type Hints

All functions should have type hints:

```python
def process_frame(
    frame: np.ndarray,
    threshold: float = 0.5,
) -> List[Detection]:
    """Process a video frame and return detections."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_dwell_time(
    entry_time: datetime,
    exit_time: datetime,
) -> float:
    """Calculate dwell time in seconds.
    
    Args:
        entry_time: When the person entered the zone.
        exit_time: When the person exited the zone.
        
    Returns:
        Dwell time in seconds.
        
    Raises:
        ValueError: If exit_time is before entry_time.
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_analytics.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Name test files `test_*.py`
- Name test functions `test_*`

Example:
```python
import pytest
from src.analytics import CustomerJourneyTracker

class TestCustomerJourneyTracker:
    def test_journey_creation(self):
        tracker = CustomerJourneyTracker(zones=[...])
        tracker.update(track_id=1, position=(0.1, 0.9))
        
        journeys = tracker.get_active_journeys()
        assert len(journeys) == 1
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation if needed

### 3. Commit Messages

Follow conventional commits:

```
feat: add customer re-identification module
fix: resolve queue counting edge case
docs: update API reference for heatmap endpoint
test: add integration tests for pipeline
refactor: simplify detection batch processing
```

### 4. Submit PR

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create Pull Request on GitHub

3. Fill in the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if UI changes)

### 5. Code Review

- Address reviewer feedback
- Keep PR focused on single feature/fix
- Rebase on main if needed

## Issue Reporting

### Bug Reports

Include:
- Environment (OS, Python version, GPU)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
- Minimal reproducible example

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered

## Project Structure

```
src/
├── vision/          # Detection and tracking
├── analytics/       # Business analytics
├── edge/            # Edge deployment
├── api/             # REST API
└── utils/           # Shared utilities
```

When adding new modules:
1. Create `__init__.py` with exports
2. Add docstring to module
3. Update parent `__init__.py`
4. Add tests

## Documentation

- Update README.md for user-facing changes
- Update API_REFERENCE.md for API changes
- Add docstrings to all public functions
- Include code examples where helpful

## Questions?

- Open a GitHub Discussion for questions
- Check existing issues before creating new ones
- Tag maintainers for urgent issues

Thank you for contributing! 🎉
