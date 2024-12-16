import pytest
import torch
from src.models.lstm_model import LSTMModel, StockPredictor
from src.config import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger("test.unit.lstm_model")

def test_lstm_model():
    config = ModelConfig()
    model = LSTMModel(config)
    
    batch_size = 32
    seq_length = config.SEQUENCE_LENGTH
    x = torch.randn(batch_size, seq_length, config.INPUT_SIZE)
    
    output = model(x)
    logger.info(f"Testing LSTM model output shape: {output.shape}")
    assert output.shape == (batch_size, config.OUTPUT_SIZE)

def test_stock_predictor():
    config = ModelConfig()
    predictor = StockPredictor(config)
    logger.info("Testing StockPredictor initialization")
    assert isinstance(predictor.model, LSTMModel)
    assert predictor.device in [torch.device('cuda'), torch.device('cpu')]
