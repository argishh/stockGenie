import pytest
import torch
from src.models.lstm_model import LSTMModel, StockPredictor
from src.config import ModelConfig

def test_lstm_model():
    config = ModelConfig()
    model = LSTMModel(config)
    
    batch_size = 32
    seq_length = config.SEQUENCE_LENGTH
    x = torch.randn(batch_size, seq_length, config.INPUT_SIZE)
    
    output = model(x)
    assert output.shape == (batch_size, config.OUTPUT_SIZE)

def test_stock_predictor():
    config = ModelConfig()
    predictor = StockPredictor(config)
    assert isinstance(predictor.model, LSTMModel)
    assert predictor.device in [torch.device('cuda'), torch.device('cpu')]
