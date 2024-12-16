import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import StockDataLoader, StockDataset
from src.config import ModelConfig

def test_stock_dataset():
    data = np.random.rand(100, 1)
    sequence_length = 10
    dataset = StockDataset(data, sequence_length)
    
    assert len(dataset) == 90
    sequence, target = dataset[0]
    assert sequence.shape == (10, 1)
    assert target.shape == (1,)

def test_stock_data_loader():
    config = ModelConfig()
    data_loader = StockDataLoader(config)
    
    data, info = data_loader.fetch_stock_data("AAPL")
    assert isinstance(data, pd.DataFrame)
    assert isinstance(info, dict)
    
    loader, scaler = data_loader.prepare_data(data)
    assert loader is not None
    assert scaler is not None
