# Contributing to Multi-Agent AI System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/multi-agent-ai-system.git
   cd multi-agent-ai-system
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/saurabh-rai/multi-agent-ai-system.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized development)
- An OpenAI API key (for testing)

### Local Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Docker Setup

```bash
docker-compose -f docker/docker-compose.yml up --build
```

## Making Changes

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes

### Workflow

1. **Create a branch** from `develop`:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits

3. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

## Pull Request Process

1. **Ensure all tests pass**:
   ```bash
   pytest
   ```

2. **Run linting**:
   ```bash
   black .
   ruff check .
   mypy src/
   ```

3. **Update documentation** if needed

4. **Create a Pull Request** with:
   - Clear title describing the change
   - Description of what and why
   - Link to related issues
   - Screenshots (if UI changes)

5. **Address review feedback** promptly

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] All CI checks pass
- [ ] Code follows project style

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these tools:
- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

### Code Organization

```
src/
├── agents/          # Agent implementations
├── api/             # FastAPI endpoints
├── core/            # Core utilities
├── rag/             # RAG pipeline
└── tools/           # Agent tools
```

### Naming Conventions

- **Classes**: PascalCase (`ResearchAgent`)
- **Functions/Methods**: snake_case (`execute_task`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_ITERATIONS`)
- **Private**: Leading underscore (`_internal_method`)

### Type Hints

Use type hints for all public functions:

```python
async def execute(
    self,
    task: str,
    context: Optional[ConversationContext] = None,
    **kwargs
) -> AgentResult:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.
    """
```

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific tests
pytest tests/unit/test_agents.py

# Verbose output
pytest -v --tb=short
```

### Writing Tests

- Place tests in `tests/` mirroring `src/` structure
- Use `pytest` fixtures for setup
- Mock external services
- Aim for >80% coverage on new code

### Test Structure

```python
class TestResearchAgent:
    """Tests for ResearchAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return ResearchAgent()

    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "research_agent"

    @pytest.mark.asyncio
    async def test_execute(self, agent):
        """Test task execution."""
        result = await agent.execute("test")
        assert result.success
```

## Documentation

### Where to Document

- **README.md**: Project overview, quick start
- **docs/**: Detailed documentation
- **Docstrings**: API documentation
- **Comments**: Complex logic explanation

### Documentation Standards

- Keep docs up-to-date with code changes
- Include code examples
- Explain the "why", not just the "what"
- Use Markdown for formatting

## Questions?

If you have questions:
1. Check existing issues
2. Search the documentation
3. Open a new issue with the `question` label

Thank you for contributing! 🎉
