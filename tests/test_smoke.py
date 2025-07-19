"""
Fast smoke tests for basic functionality.
These tests run very quickly to catch import and basic setup issues.
"""
import pytest


def test_timecopilot_imports():
    """Test that main classes can be imported without errors."""
    from timecopilot import TimeCopilot, TimeCopilotForecaster
    assert TimeCopilot is not None
    assert TimeCopilotForecaster is not None


def test_models_import():
    """Test that model classes can be imported."""
    from timecopilot.models import SeasonalNaive, ZeroModel
    assert SeasonalNaive is not None
    assert ZeroModel is not None


def test_basic_model_creation():
    """Test that basic models can be instantiated."""
    from timecopilot.models import SeasonalNaive, ZeroModel
    
    model1 = SeasonalNaive()
    model2 = ZeroModel()
    
    assert model1 is not None
    assert model2 is not None
    assert hasattr(model1, 'forecast')
    assert hasattr(model2, 'forecast')


def test_forecaster_creation():
    """Test that TimeCopilotForecaster can be created with basic models."""
    from timecopilot.forecaster import TimeCopilotForecaster
    from timecopilot.models import SeasonalNaive, ZeroModel
    
    models = [SeasonalNaive(), ZeroModel()]
    forecaster = TimeCopilotForecaster(models=models)
    
    assert forecaster is not None
    assert len(forecaster.models) == 2