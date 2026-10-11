# Contributing to GPU-Accelerated ML Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 12.x
- TensorRT 8.6+
- Python 3.10+
- Docker (optional)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/saurabh-rai/gpu-ml-pipeline.git
cd gpu-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Build CUDA kernels
cd src/cuda && python setup.py develop && cd ../..

# Run tests
pytest tests/ -v
```

## Development Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `perf/` - Performance improvements
- `docs/` - Documentation updates

### Commit Messages

Use conventional commit format:
```
type(scope): description

[optional body]
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `chore`

Examples:
```
feat(cuda): add fused preprocessing kernel
fix(tensorrt): handle dynamic batch sizes
perf(inference): optimize memory allocation
```

## Code Standards

### Python

- Format with `black` (line length 100)
- Sort imports with `isort`
- Lint with `ruff`
- Type hints required for public APIs

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### CUDA/C++

- Follow NVIDIA coding guidelines
- Use `__restrict__` for pointer arguments
- Document kernel parameters
- Include performance characteristics in comments

### Documentation

- Docstrings for all public functions/classes
- Include usage examples
- Document performance characteristics

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# GPU tests (requires GPU)
pytest tests/ -v -m gpu
```

### Writing Tests

- Test files: `test_*.py`
- Use pytest fixtures
- Mark GPU-required tests with `@pytest.mark.gpu`
- Include benchmarks with `@pytest.mark.benchmark`

## Performance Guidelines

### CUDA Kernels

1. Profile before optimizing
2. Maximize memory coalescing
3. Minimize warp divergence
4. Use shared memory for frequently accessed data
5. Document throughput in kernel comments

### TensorRT

1. Use appropriate precision (FP16/INT8)
2. Enable timing cache for faster rebuilds
3. Profile with `trtexec` before deployment
4. Document accuracy vs. speed tradeoffs

## Pull Request Process

1. Create feature branch from `main`
2. Make changes with tests
3. Update documentation
4. Run full test suite
5. Submit PR with description
6. Address review feedback
7. Squash and merge

### PR Checklist

- [ ] Tests pass locally
- [ ] Code formatted and linted
- [ ] Documentation updated
- [ ] Performance impact documented (if applicable)
- [ ] CUDA kernels tested on target GPU

## Questions?

Open an issue or contact the maintainers.
