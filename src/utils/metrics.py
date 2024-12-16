# src/utils/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
from ..utils.logger import setup_logger

logger = setup_logger("metrics")

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    try:
        metrics = {
            'MSE': round(mean_squared_error(actual, predicted), 2),
            'RMSE': round(np.sqrt(mean_squared_error(actual, predicted)), 2),
            'MAE': round(mean_absolute_error(actual, predicted), 2),
            'R2': round(r2_score(actual, predicted), 2)
        }
        
        logger.info("Metrics calculated successfully")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise
