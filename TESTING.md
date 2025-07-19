# Testing Guide

This project uses a two-tier testing strategy to balance speed and comprehensive coverage.

## Test Categories

### Fast Tests (Regular CI)
- **Purpose**: Quick validation for every PR/push
- **Runtime**: ~2-3 minutes
- **Models**: Lightweight benchmark models (SeasonalNaive, ZeroModel, AutoARIMA, Prophet, etc.)
- **Trigger**: Runs on every push/PR
- **Command**: `uv run pytest` (excludes slow tests by default)

### Slow Tests (Comprehensive)
- **Purpose**: Full model validation including heavy foundation models  
- **Runtime**: ~30-45 minutes
- **Models**: All models including Chronos, Moirai, TimesFM, TiRex, etc.
- **Trigger**: Nightly schedule + manual dispatch
- **Command**: `uv run pytest -m slow`

### Live Tests
- **Purpose**: Test integration with external APIs (OpenAI, etc.)
- **Models**: Uses real LLM endpoints
- **Trigger**: Regular CI (with API keys)
- **Command**: `uv run pytest -m live`

## Running Tests Locally

```bash
# Fast tests only (recommended for development)
uv run pytest

# Include slow foundation model tests  
uv run pytest -m "not live"

# All tests including live API calls
uv run pytest -m ""

# Specific test categories
uv run pytest -m slow          # Foundation models only
uv run pytest -m live          # API integration only
uv run pytest tests/test_smoke.py  # Basic smoke tests
```

## Test Optimization Details

### Model Selection Strategy
- **Fast models** (`tests/conftest.py::fast_models`): Used in regular CI
  - Benchmark models: SeasonalNaive, ZeroModel, AutoARIMA, Prophet, ADIDA
  - Lightweight foundation: TabPFN (MOCK mode)
  
- **Comprehensive models** (`tests/conftest.py::all_models`): Used in nightly tests
  - All fast models plus heavy foundation models
  - Foundation models: Chronos, Moirai, TimesFM, TiRex, Toto

### Parameterization Reductions
To speed up CI, several test parameters were reduced:
- **Frequencies**: 4 → 2 (H,D,W-MON,MS → D,W-MON)
- **Series counts**: 5 → 3 per test
- **Quantiles**: 9 → 3 (0.1-0.9 → 0.1,0.5,0.9)  
- **Cross-validation horizon**: 12 → 6
- **Python versions**: 4 → 2 (3.11, 3.12 for regular CI)

### Caching Strategy
- **Model weights**: HuggingFace and PyTorch caches preserved between runs
- **Dependencies**: UV package cache enabled
- **Cache keys**: Include Python version and dependency hash

## Adding New Tests

### For Fast Models
Add tests to existing files in `tests/` - they will automatically use fast models and run in regular CI.

### For Foundation Models
- Add tests with `@pytest.mark.slow` decorator
- Consider using reduced parameterization for speed
- Test will run in nightly comprehensive workflow

### Example
```python
@pytest.mark.slow
@pytest.mark.parametrize("model", all_models)  # Use all_models for comprehensive
def test_new_foundation_feature(model):
    # Test with foundation models
    pass

@pytest.mark.parametrize("model", models)  # Use models for fast tests
def test_new_basic_feature(model):
    # Test with fast models only
    pass
```

## CI Workflows

### `.github/workflows/ci.yaml` 
- **Triggers**: Push to main, PRs
- **Python versions**: 3.11, 3.12
- **Timeout**: 10 minutes regular tests, 15 minutes live tests
- **Excludes**: Slow tests (foundation models)

### `.github/workflows/comprehensive.yaml`
- **Triggers**: Nightly at 2 AM UTC, manual dispatch, model file changes
- **Python versions**: 3.11, 3.12  
- **Timeout**: 45 minutes
- **Includes**: Only slow tests (foundation models)

This structure ensures rapid feedback for developers while maintaining comprehensive test coverage through automated nightly runs.