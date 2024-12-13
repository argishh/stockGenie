import logging
from datetime import datetime
from pathlib import Path
from src.config import AppConfig

def setup_logger(name: str) -> logging.Logger:
    config = AppConfig()
    
    # Map logger names to their directories
    logger_dir_mapping = {
        "streamlit_app": config.LOG_DIRS["app"],
        "lstm_model": config.LOG_DIRS["model"],
        "data_loader": config.LOG_DIRS["data"],
        "metrics": config.LOG_DIRS["metrics"],
        "model_optimization": config.LOG_DIRS["optimization"],
        # Default to app logs if not specified
        "default": config.LOG_DIRS["app"]
    }
    
    # Get the appropriate log directory
    log_dir = Path(logger_dir_mapping.get(name, logger_dir_mapping["default"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler with appropriate directory
    file_handler = logging.FileHandler(
        log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
