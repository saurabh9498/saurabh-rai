"""
Neural Network Fraud Classifier

PyTorch neural network optimized for:
- Sequential pattern recognition
- Embedding categorical features
- Handling class imbalance with focal loss
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class NeuralNetConfig:
    """Neural network configuration."""
    input_dim: int = 17
    hidden_dims: List[int] = None
    dropout: float = 0.3
    batch_norm: bool = True
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10
    focal_loss_gamma: float = 2.0  # For class imbalance
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class FraudNeuralNetwork(nn.Module):
    """
    Neural network for fraud detection.
    
    Architecture:
    - Input layer with batch normalization
    - Multiple hidden layers with dropout
    - Sigmoid output for probability
    """
    
    def __init__(self, config: NeuralNetConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = config.input_dim
        
        # Input batch normalization
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(prev_dim))
            
        # Hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class NeuralNetFraudModel:
    """
    PyTorch neural network for fraud detection.
    
    Strengths:
    - Captures non-linear patterns
    - Handles sequential features via embeddings
    - Focal loss for class imbalance
    """
    
    def __init__(
        self,
        config: Optional[NeuralNetConfig] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.config = config or NeuralNetConfig()
        self.feature_names = feature_names
        self.model: Optional[FraudNeuralNetwork] = None
        self.device = torch.device(self.config.device) if TORCH_AVAILABLE else None
        self._training_history: List[Dict[str, float]] = []
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        logger.info("Training Neural Network fraud classifier...")
        
        # Update input dimension
        self.config.input_dim = X_train.shape[1]
        
        # Create model
        self.model = FraudNeuralNetwork(self.config).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
            )
            
        # Loss and optimizer
        criterion = FocalLoss(gamma=self.config.focal_loss_gamma)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        val_loss += criterion(outputs, y_batch).item()
                        
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            self._training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
        metrics = {
            "epochs_trained": epoch + 1,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_loss,
        }
        
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get fraud probability.
        
        Args:
            X: Features array
            
        Returns:
            Fraud probabilities (0-1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
            
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            
        return outputs.cpu().numpy()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def save(self, path: str):
        """Save model to disk."""
        if not TORCH_AVAILABLE:
            return
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "feature_names": self.feature_names,
            "training_history": self._training_history,
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "NeuralNetFraudModel":
        """Load model from disk."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        checkpoint = torch.load(path)
        
        instance = cls(
            config=checkpoint["config"],
            feature_names=checkpoint["feature_names"],
        )
        
        instance.model = FraudNeuralNetwork(instance.config).to(instance.device)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance._training_history = checkpoint["training_history"]
        
        logger.info(f"Model loaded from {path}")
        return instance


class MockNeuralNetModel(NeuralNetFraudModel):
    """Mock neural network for testing."""
    
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        return {"epochs_trained": 50, "best_val_loss": 0.05}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        return np.random.beta(1, 15, n_samples)
