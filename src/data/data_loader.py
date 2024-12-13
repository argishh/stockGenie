from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from src.config import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

class StockDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = np.array(self.data[idx:idx + self.sequence_length], dtype=np.float32)
        target = np.array(self.data[idx + self.sequence_length], dtype=np.float32)
        
        # Reshape sequence to [sequence_length, features]
        sequence = sequence.reshape(self.sequence_length, 1)
        # Reshape target to [1] for loss calculation
        target = target.reshape(1)
        
        return torch.from_numpy(sequence), torch.from_numpy(target)

class StockDataLoader:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def fetch_stock_data(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        try:
            logger.info(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            data = stock.history(period="5y")
            # Set the ticker as name attribute
            data.name = ticker
            return data, stock.info
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None, None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[DataLoader, MinMaxScaler]:
        try:
            logger.info("Preparing data for training")
            
            if data is None or data.empty:
                raise ValueError("No data available for processing")
            
            if len(data) < self.config.SEQUENCE_LENGTH:
                raise ValueError(f"Insufficient data: need at least {self.config.SEQUENCE_LENGTH} points")
            
            # Ensure data is correctly shaped
            close_prices = data['Close'].values.reshape(-1, 1)
            logger.info(f"Data shape before scaling: {close_prices.shape}")
            
            scaled_data = self.scaler.fit_transform(close_prices)
            logger.info(f"Data shape after scaling: {scaled_data.shape}")
            
            dataset = StockDataset(scaled_data, self.config.SEQUENCE_LENGTH)
            logger.info(f"Dataset size: {len(dataset)}")
            
            dataloader = DataLoader(
                dataset,
                batch_size=min(self.config.BATCH_SIZE, len(dataset)),
                shuffle=True,
                drop_last=False
            )
            logger.info(f"Number of batches: {len(dataloader)}")
            
            return dataloader, self.scaler
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
