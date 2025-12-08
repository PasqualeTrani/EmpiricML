# base imports 
from abc import ABC, abstractmethod

# data wranglers 
import polars as pl
import numpy as np

# deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------------------------------------
# DEFINITION OF THE ABSTRACT BASE CLASS (Copied for dependency self-containment) 
# ------------------------------------------------------------------------------------------

class BaseRegressor(ABC):
    @abstractmethod
    def fit(self, df : pl.LazyFrame):
        pass
    
    @abstractmethod
    def predict(self, df : pl.LazyFrame):
        pass

# ------------------------------------------------------------------------------------------
# PYTORCH MODEL DEFINITION
# ------------------------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """
    A basic Multi-Layer Perceptron for regression.
    """
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# ------------------------------------------------------------------------------------------
# PYTORCH REGRESSOR IMPLEMENTATION (SKLEARN-STYLE WRAPPER)
# ------------------------------------------------------------------------------------------

class mlp_reg(BaseRegressor):
    """
    A PyTorch MLP wrapped to expose the scikit-learn-style fit and predict 
    interface using Polars LazyFrames for regression tasks.
    """
    def __init__(self, features: list[str], target: str, 
                 hidden_size: int = 64, 
                 epochs: int = 10, 
                 batch_size: int = 32, 
                 learning_rate: float = 0.001,
                 **mlp_kwargs):
        
        # Data/Feature parameters
        self.features = features
        self.target = target
        self.input_size = len(features)
        
        # MLP hyperparameters
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Internal PyTorch components
        self.model = SimpleMLP(self.input_size, self.hidden_size, output_size=1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, **mlp_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        """
        Trains the PyTorch MLP model.
        """
        # 1. Data Preparation: Polars -> NumPy -> PyTorch Tensors
        X_np = lf.select(self.features).collect().to_numpy(ordered=True).astype(np.float32)
        y_np = lf.select(self.target).collect().to_series().to_numpy(ordered=True).astype(np.float32)
        
        X_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y_np).unsqueeze(1) # Add dimension for regression target
        
        # 2. PyTorch Dataset and DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in data_loader:
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                y_pred = self.model(X_batch)
                
                # Compute loss
                loss = self.loss_fn(y_pred, y_batch)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
        return self

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        """
        Makes predictions using the trained PyTorch MLP model.
        """
        # 1. Data Preparation: Polars -> NumPy -> PyTorch Tensor
        X_np = lf.select(self.features).collect().to_numpy(ordered=True).astype(np.float32)
        X_tensor = torch.from_numpy(X_np)
        
        # 2. Prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # 3. Convert back to NumPy array (for scikit-learn compatibility)
        return predictions.numpy().flatten()