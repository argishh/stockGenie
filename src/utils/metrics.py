# src/utils/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
from ..utils.logger import setup_logger

logger = setup_logger("metrics")

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    try:
        metrics = {
            'MSE': mean_squared_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'R2': r2_score(actual, predicted)
        }
        
        logger.info("Metrics calculated successfully")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise
