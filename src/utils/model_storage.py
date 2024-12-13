import os
import glob
from pathlib import Path
import torch
from typing import Optional
from src.config import AppConfig
from src.utils.logger import setup_logger

logger = setup_logger("model_storage")

class ModelStorage:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_dir = Path(config.MODEL_DIR)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, model: torch.nn.Module, ticker: str) -> None:
        # Get existing versions
        pattern = self.config.MODEL_FILENAME_TEMPLATE.format(ticker=ticker, version='*')
        existing_files = sorted(self.model_dir.glob(pattern))
        
        # Remove oldest if max versions reached
        while len(existing_files) >= self.config.MAX_MODEL_VERSIONS:
            oldest_file = existing_files.pop(0)
            oldest_file.unlink()
            logger.info(f"Removed old model version: {oldest_file.name}")
        
        # Save new version
        version = len(existing_files) + 1
        filename = self.config.MODEL_FILENAME_TEMPLATE.format(ticker=ticker, version=version)
        filepath = self.model_dir / filename
        
        torch.save(model.state_dict(), filepath)
        logger.info(f"Saved model: {filepath}")
    
    def load_latest_model(self, model: torch.nn.Module, ticker: str) -> Optional[torch.nn.Module]:
        pattern = self.config.MODEL_FILENAME_TEMPLATE.format(ticker=ticker, version='*')
        existing_files = sorted(self.model_dir.glob(pattern))
        
        if not existing_files:
            return None
        
        latest_model = existing_files[-1]
        model.load_state_dict(torch.load(latest_model))
        logger.info(f"Loaded model: {latest_model.name}")
        return model