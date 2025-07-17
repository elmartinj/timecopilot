#!/usr/bin/env python3
"""
Test for the Toto batch size fix.
This test validates that the TimeSeriesDataset handles batch sizes correctly.
"""

import pytest
import torch
from timecopilot.models.foundational.utils import TimeSeriesDataset


def test_timeseries_dataset_batch_size_handling():
    """Test that TimeSeriesDataset correctly handles cases where n_series > batch_size."""
    
    # Test with more series than batch size
    n_series = 25
    batch_size = 16
    
    # Create mock tensors
    tensors = [torch.randn(50, dtype=torch.bfloat16) for _ in range(n_series)]
    uids = [f"ts_{i}" for i in range(n_series)]
    last_times = [f"2023-01-{i+1:02d}" for i in range(n_series)]
    
    # Create dataset
    dataset = TimeSeriesDataset(tensors, uids, last_times, batch_size)
    
    # Verify dataset properties
    expected_batches = n_series // batch_size + (1 if n_series % batch_size > 0 else 0)
    assert len(dataset) == expected_batches
    
    # Verify batch sizes
    batches = list(dataset)
    assert len(batches) == 2  # 25 series / 16 batch_size = 2 batches
    assert len(batches[0]) == 16  # First batch full
    assert len(batches[1]) == 9   # Second batch partial
    
    # Verify total series count
    total_series_in_batches = sum(len(batch) for batch in batches)
    assert total_series_in_batches == n_series


def test_timeseries_dataset_exact_batch_size():
    """Test that TimeSeriesDataset handles exact multiples of batch size."""
    
    # Test with exact multiple of batch size
    n_series = 32
    batch_size = 16
    
    # Create mock tensors
    tensors = [torch.randn(50, dtype=torch.bfloat16) for _ in range(n_series)]
    uids = [f"ts_{i}" for i in range(n_series)]
    last_times = [f"2023-01-{i+1:02d}" for i in range(n_series)]
    
    # Create dataset
    dataset = TimeSeriesDataset(tensors, uids, last_times, batch_size)
    
    # Verify dataset properties
    expected_batches = n_series // batch_size
    assert len(dataset) == expected_batches
    
    # Verify batch sizes
    batches = list(dataset)
    assert len(batches) == 2  # 32 series / 16 batch_size = 2 batches
    assert len(batches[0]) == 16  # First batch full
    assert len(batches[1]) == 16  # Second batch full
    
    # Verify total series count
    total_series_in_batches = sum(len(batch) for batch in batches)
    assert total_series_in_batches == n_series


def test_timeseries_dataset_small_batch_size():
    """Test that TimeSeriesDataset handles small batch sizes correctly."""
    
    # Test with small batch size
    n_series = 10
    batch_size = 3
    
    # Create mock tensors
    tensors = [torch.randn(50, dtype=torch.bfloat16) for _ in range(n_series)]
    uids = [f"ts_{i}" for i in range(n_series)]
    last_times = [f"2023-01-{i+1:02d}" for i in range(n_series)]
    
    # Create dataset
    dataset = TimeSeriesDataset(tensors, uids, last_times, batch_size)
    
    # Verify dataset properties
    expected_batches = n_series // batch_size + (1 if n_series % batch_size > 0 else 0)
    assert len(dataset) == expected_batches
    
    # Verify batch sizes
    batches = list(dataset)
    assert len(batches) == 4  # 10 series / 3 batch_size = 4 batches
    assert len(batches[0]) == 3  # First batch full
    assert len(batches[1]) == 3  # Second batch full
    assert len(batches[2]) == 3  # Third batch full
    assert len(batches[3]) == 1  # Fourth batch partial
    
    # Verify total series count
    total_series_in_batches = sum(len(batch) for batch in batches)
    assert total_series_in_batches == n_series


if __name__ == "__main__":
    pytest.main([__file__, "-v"])