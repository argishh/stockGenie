import pytest
import numpy as np
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

logger = setup_logger("test.unit.metrics")

def test_metrics_calculation():
    """Test metrics calculation with known values"""
    actual = np.array([1, 2, 3, 4, 5])
    predicted = np.array([1.1, 2.1, 2.9, 4.2, 5.1])
    
    metrics = calculate_metrics(actual, predicted)
    
    logger.info("Testing metrics calculation with known values")
    logger.info(f"Metrics: {metrics}")
    
    assert "MSE" in metrics
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "R2" in metrics
    assert metrics["R2"] > 0.95  # Should be very good for these values

def test_metrics_error_handling():
    """Test metrics calculation error handling"""
    with pytest.raises(Exception):
        calculate_metrics([], [1, 2, 3])  # Different lengths