import pytest
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.config import ModelConfig
from src.models.lstm_model import StockPredictor, EarlyStopping  # Import EarlyStopping
from src.data.data_loader import StockDataLoader
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

logger = setup_logger("model_optimization")

@pytest.fixture
def data_loader():
    return StockDataLoader(ModelConfig())

@pytest.fixture
def stock_data(data_loader):
    data, _ = data_loader.fetch_stock_data("AAPL")
    if data is None or data.empty:
        pytest.skip("Could not fetch test data")
    return data['Close'].values

def evaluate_model(predictor, train_data, test_data, scaler):
    """Helper function to evaluate model performance"""
    # Train the model
    train_losses, val_losses = predictor.train_with_validation(train_data, test_data)
    
    # Make predictions on test data
    scaled_test = scaler.transform(test_data.reshape(-1, 1))
    predictions = predictor.predict(scaled_test, scaler)
    
    # Calculate metrics
    metrics = calculate_metrics(test_data[-len(predictions):], predictions.flatten())
    
    return {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'metrics': metrics
    }

def test_different_architectures(stock_data, data_loader):
    """Test different model architectures"""
    # Test configurations
    configs = [
        {"HIDDEN_SIZE": 32, "NUM_LAYERS": 2},
        {"HIDDEN_SIZE": 64, "NUM_LAYERS": 2},
        {"HIDDEN_SIZE": 128, "NUM_LAYERS": 3},
        {"HIDDEN_SIZE": 64, "NUM_LAYERS": 4},
    ]
    
    # Use TimeSeriesSplit for proper validation
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    
    for config_vars in configs:
        config_results = []
        logger.info(f"\nTesting architecture: {config_vars}")
        
        # Create new config with test values
        test_config = ModelConfig()
        for key, value in config_vars.items():
            setattr(test_config, key, value)
        
        # Test on multiple splits
        for fold, (train_idx, test_idx) in enumerate(tscv.split(stock_data)):
            train_data = stock_data[train_idx]
            test_data = stock_data[test_idx]
            
            # Scale the data
            scaler = data_loader.scaler
            scaler.fit(train_data.reshape(-1, 1))
            
            predictor = StockPredictor(test_config)
            fold_results = evaluate_model(predictor, train_data, test_data, scaler)
            
            logger.info(f"Fold {fold + 1} results:")
            logger.info(f"Train Loss: {fold_results['final_train_loss']:.4f}")
            logger.info(f"Val Loss: {fold_results['final_val_loss']:.4f}")
            logger.info(f"RMSE: {fold_results['metrics']['RMSE']:.4f}")
            logger.info(f"R2: {fold_results['metrics']['R2']:.4f}")
            
            config_results.append(fold_results)
        
        # Average results across folds
        avg_results = {
            "config": config_vars,
            "avg_train_loss": np.mean([r['final_train_loss'] for r in config_results]),
            "avg_val_loss": np.mean([r['final_val_loss'] for r in config_results]),
            "avg_rmse": np.mean([r['metrics']['RMSE'] for r in config_results]),
            "avg_r2": np.mean([r['metrics']['R2'] for r in config_results])
        }
        
        results.append(avg_results)
        logger.info(f"\nAverage results for {config_vars}:")
        logger.info(f"Avg Train Loss: {avg_results['avg_train_loss']:.4f}")
        logger.info(f"Avg Val Loss: {avg_results['avg_val_loss']:.4f}")
        logger.info(f"Avg RMSE: {avg_results['avg_rmse']:.4f}")
        logger.info(f"Avg R2: {avg_results['avg_r2']:.4f}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['avg_val_loss'])
    logger.info("\nBest architecture configuration:")
    logger.info(f"Config: {best_config['config']}")
    logger.info(f"Validation Loss: {best_config['avg_val_loss']:.4f}")
    logger.info(f"RMSE: {best_config['avg_rmse']:.4f}")
    logger.info(f"R2: {best_config['avg_r2']:.4f}")
    
    # Add assertions instead of returning results
    assert len(results) > 0
    assert best_config['avg_val_loss'] < float('inf')
    assert best_config['avg_rmse'] > 0
    for result in results:
        assert result['avg_train_loss'] > 0
        assert result['avg_val_loss'] > 0

def test_regularization_methods(stock_data, data_loader):
    """Test different regularization methods"""
    configs = [
        {"DROPOUT": 0.1},
        {"DROPOUT": 0.2},
        {"DROPOUT": 0.3},
        {"DROPOUT": 0.4},
    ]
    
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    
    for config_vars in configs:
        config_results = []
        logger.info(f"\nTesting regularization: {config_vars}")
        
        test_config = ModelConfig()
        for key, value in config_vars.items():
            setattr(test_config, key, value)
        
        # Test on multiple splits
        for fold, (train_idx, test_idx) in enumerate(tscv.split(stock_data)):
            train_data = stock_data[train_idx]
            test_data = stock_data[test_idx]
            
            # Scale the data
            scaler = data_loader.scaler
            scaler.fit(train_data.reshape(-1, 1))
            
            predictor = StockPredictor(test_config)
            fold_results = evaluate_model(predictor, train_data, test_data, scaler)
            
            logger.info(f"Fold {fold + 1} results:")
            logger.info(f"Train Loss: {fold_results['final_train_loss']:.4f}")
            logger.info(f"Val Loss: {fold_results['final_val_loss']:.4f}")
            logger.info(f"RMSE: {fold_results['metrics']['RMSE']:.4f}")
            logger.info(f"R2: {fold_results['metrics']['R2']:.4f}")
            
            config_results.append(fold_results)
        
        # Average results across folds
        avg_results = {
            "config": config_vars,
            "avg_train_loss": np.mean([r['final_train_loss'] for r in config_results]),
            "avg_val_loss": np.mean([r['final_val_loss'] for r in config_results]),
            "avg_rmse": np.mean([r['metrics']['RMSE'] for r in config_results]),
            "avg_r2": np.mean([r['metrics']['R2'] for r in config_results])
        }
        
        results.append(avg_results)
        logger.info(f"\nAverage results for {config_vars}:")
        logger.info(f"Avg Train Loss: {avg_results['avg_train_loss']:.4f}")
        logger.info(f"Avg Val Loss: {avg_results['avg_val_loss']:.4f}")
        logger.info(f"Avg RMSE: {avg_results['avg_rmse']:.4f}")
        logger.info(f"Avg R2: {avg_results['avg_r2']:.4f}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['avg_val_loss'])
    logger.info("\nBest architecture configuration:")
    logger.info(f"Config: {best_config['config']}")
    logger.info(f"Validation Loss: {best_config['avg_val_loss']:.4f}")
    logger.info(f"RMSE: {best_config['avg_rmse']:.4f}")
    logger.info(f"R2: {best_config['avg_r2']:.4f}")
    
    # Add assertions
    assert len(results) > 0
    assert best_config['avg_val_loss'] < float('inf')
    for result in results:
        assert result['avg_train_loss'] > 0
        assert result['avg_val_loss'] > 0

def test_early_stopping(stock_data, data_loader):
    """Test early stopping implementation"""
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    
    logger.info("\nTesting early stopping with different patience values")
    patience_values = [3, 5, 7, 10]
    
    for patience in patience_values:
        logger.info(f"\nTesting with patience: {patience}")
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(stock_data)):
            train_data = stock_data[train_idx]
            test_data = stock_data[test_idx]
            
            # Scale the data
            scaler = data_loader.scaler
            scaler.fit(train_data.reshape(-1, 1))
            
            # Initialize model and early stopping
            predictor = StockPredictor(ModelConfig())
            early_stopping = EarlyStopping(patience=patience)  # Use imported class
            
            # Train with early stopping
            train_losses, val_losses = predictor.train_with_validation(train_data, test_data)
            
            fold_results.append({
                'epochs_trained': len(train_losses),
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'stopped_early': len(train_losses) < ModelConfig().EPOCHS
            })
            
            logger.info(f"Fold {fold + 1} Results:")
            logger.info(f"Epochs trained: {len(train_losses)}")
            logger.info(f"Final train loss: {train_losses[-1]:.4f}")
            logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
            logger.info(f"Stopped early: {fold_results[-1]['stopped_early']}")
        
        # Average results for this patience value
        avg_results = {
            'patience': patience,
            'avg_epochs': np.mean([r['epochs_trained'] for r in fold_results]),
            'avg_train_loss': np.mean([r['final_train_loss'] for r in fold_results]),
            'avg_val_loss': np.mean([r['final_val_loss'] for r in fold_results]),
            'early_stop_rate': np.mean([r['stopped_early'] for r in fold_results])
        }
        
        results.append(avg_results)
        logger.info(f"\nAverage results for patience={patience}:")
        logger.info(f"Avg epochs trained: {avg_results['avg_epochs']:.1f}")
        logger.info(f"Avg train loss: {avg_results['avg_train_loss']:.4f}")
        logger.info(f"Avg validation loss: {avg_results['avg_val_loss']:.4f}")
        logger.info(f"Early stopping rate: {avg_results['early_stop_rate']*100:.1f}%")
    
    # Find best patience value
    best_result = min(results, key=lambda x: x['avg_val_loss'])
    logger.info("\nBest early stopping configuration:")
    logger.info(f"Patience: {best_result['patience']}")
    logger.info(f"Avg epochs needed: {best_result['avg_epochs']:.1f}")
    logger.info(f"Avg validation loss: {best_result['avg_val_loss']:.4f}")
    
    # Add assertions
    assert len(results) > 0
    assert best_result['avg_val_loss'] < float('inf')
    assert best_result['avg_epochs'] <= ModelConfig().EPOCHS
    for result in results:
        assert result['avg_train_loss'] > 0
        assert result['avg_val_loss'] > 0
        assert 0 <= result['early_stop_rate'] <= 1

if __name__ == "__main__":
    # Run tests and print detailed results
    pytest.main([__file__, "-v", "--capture=no"])