import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import StockDataLoader, StockDataset
from src.config import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger("test.unit.data_loader")

def test_stock_dataset():
    data = np.random.rand(100, 1)
    sequence_length = 10
    dataset = StockDataset(data, sequence_length)
    
    logger.info("Testing StockDataset length and shapes")
    assert len(dataset) == 90
    sequence, target = dataset[0]
    assert sequence.shape == (10, 1)
    assert target.shape == (1,)

def test_stock_data_loader():
    config = ModelConfig()
    data_loader = StockDataLoader(config)
    
    data, info = data_loader.fetch_stock_data("AAPL")
    logger.info("Testing StockDataLoader fetch_stock_data")
    assert isinstance(data, pd.DataFrame)
    assert isinstance(info, dict)
    
    loader, scaler = data_loader.prepare_data(data)
    logger.info("Testing StockDataLoader prepare_data")
    assert loader is not None
    assert scaler is not None
