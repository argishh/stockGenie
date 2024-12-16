import pytest
from src.data.data_loader import StockDataLoader
from src.models.lstm_model import StockPredictor
from src.config import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger("test.integration.end_to_end")

def test_full_prediction_pipeline():
    """Test the entire prediction pipeline"""
    config = ModelConfig()
    data_loader = StockDataLoader(config)
    predictor = StockPredictor(config)
    
    logger.info("Starting end-to-end test")
    
    # Fetch data
    data, info = data_loader.fetch_stock_data("AAPL")
    assert data is not None
    
    # Prepare data
    train_loader, scaler = data_loader.prepare_data(data)
    assert train_loader is not None
    
    # Train model
    losses = predictor.train(train_loader)
    assert len(losses) > 0
    assert losses[-1] < losses[0]  # Should improve
    
    # Make predictions
    scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
    predictions = predictor.predict(scaled_data, scaler)
    assert len(predictions) == config.PREDICTION_DAYS