import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from stqdm import stqdm

from src.config import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger("lstm_model")

class LSTMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.HIDDEN_SIZE
        self.num_layers = config.NUM_LAYERS
        
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            batch_first=True
        )
        
        self.fc = nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class StockPredictor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(config).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE
        )
        
    def train(self, train_loader: DataLoader) -> list:
        try:
            if len(train_loader) == 0:
                raise ValueError("Training data is empty")
            
            logger.info(f"Starting model training on {self.device}")
            logger.info(f"Batch size: {train_loader.batch_size}")
            losses = []
            
            self.model.train()  # Set model to training mode
            
            for epoch in stqdm(range(self.config.EPOCHS)):
                epoch_loss = 0
                batch_count = 0
                
                for sequences, targets in train_loader:
                    if sequences.size(0) == 0:
                        continue
                    
                    # Log shapes for debugging
                    logger.debug(f"Sequence shape: {sequences.shape}, Target shape: {targets.shape}")
                    
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    try:
                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, targets)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                        
                    except RuntimeError as e:
                        logger.error(f"Error in batch: {str(e)}")
                        continue
                
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    losses.append(avg_loss)
                    logger.info(f"Epoch [{epoch+1}/{self.config.EPOCHS}], Loss: {avg_loss:.4f}")
                
            return losses
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}")

    def predict(self, data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        # Reshape input sequence to match training dimensions [batch, sequence, features]
        current_sequence = torch.FloatTensor(data[-self.config.SEQUENCE_LENGTH:]).reshape(-1, 1).to(self.device)
        
        for _ in range(self.config.PREDICTION_DAYS):
            with torch.no_grad():
                # Add batch dimension and ensure correct shape
                input_seq = current_sequence.unsqueeze(0)  # [1, sequence, features]
                predicted = self.model(input_seq)
                predictions.append(predicted.cpu().numpy()[0, 0])
                
                # Update sequence: remove first element and append prediction
                current_sequence = torch.cat([
                    current_sequence[1:],
                    predicted.reshape(1, 1)  # Ensure prediction has shape [1, 1]
                ], dim=0)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        return predictions

    def train_with_validation(self, train_data: np.ndarray, val_data: np.ndarray) -> list:
        """Train with validation data to monitor overfitting"""
        train_losses = []
        val_losses = []
        early_stopping = EarlyStopping(patience=5)  # Use the class directly
        
        for epoch in range(self.config.EPOCHS):
            # Train epoch
            train_loss = self.train_epoch(train_data)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_data)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch [{epoch+1}/{self.config.EPOCHS}], "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info("Early stopping triggered")
                break
        
        return train_losses, val_losses
    
    def validate(self, val_data: np.ndarray) -> float:
        """Calculate validation loss"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        # Create sequences from validation data
        sequences = []
        targets = []
        for i in range(len(val_data) - self.config.SEQUENCE_LENGTH):
            seq = val_data[i:(i + self.config.SEQUENCE_LENGTH)]
            target = val_data[i + self.config.SEQUENCE_LENGTH]
            sequences.append(seq)
            targets.append(target)
        
        # Convert lists to numpy arrays first, then to tensors
        sequences = np.array(sequences).reshape(-1, self.config.SEQUENCE_LENGTH, 1)
        targets = np.array(targets).reshape(-1, 1)
        
        # Convert to tensors
        sequences = torch.from_numpy(sequences).float().to(self.device)
        targets = torch.from_numpy(targets).float().to(self.device)
        
        with torch.no_grad():
            # Split into batches
            num_batches = len(sequences) // self.config.BATCH_SIZE
            for i in range(num_batches):
                start_idx = i * self.config.BATCH_SIZE
                end_idx = start_idx + self.config.BATCH_SIZE
                
                batch_sequences = sequences[start_idx:end_idx]
                batch_targets = targets[start_idx:end_idx]
                
                outputs = self.model(batch_sequences)
                loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else float('inf')

    def train_epoch(self, data: np.ndarray) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        # Create sequences from data
        sequences = []
        targets = []
        for i in range(len(data) - self.config.SEQUENCE_LENGTH):
            seq = data[i:(i + self.config.SEQUENCE_LENGTH)]
            target = data[i + self.config.SEQUENCE_LENGTH]
            sequences.append(seq)
            targets.append(target)
        
        # Convert lists to numpy arrays first, then to tensors
        sequences = np.array(sequences).reshape(-1, self.config.SEQUENCE_LENGTH, 1)
        targets = np.array(targets).reshape(-1, 1)
        
        # Convert to tensors
        sequences = torch.from_numpy(sequences).float().to(self.device)
        targets = torch.from_numpy(targets).float().to(self.device)
        
        # Split into batches
        num_batches = len(sequences) // self.config.BATCH_SIZE
        for i in range(num_batches):
            start_idx = i * self.config.BATCH_SIZE
            end_idx = start_idx + self.config.BATCH_SIZE
            
            batch_sequences = sequences[start_idx:end_idx]
            # Fix: Use the same indices for targets as sequences
            batch_targets = targets[start_idx:end_idx]  # Changed from end_idx:end_idx
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_sequences)
            
            # Ensure shapes match for loss calculation
            loss = self.criterion(outputs, batch_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else float('inf')
