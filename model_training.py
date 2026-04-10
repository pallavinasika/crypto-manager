import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional
import pickle
from pathlib import Path

# Fix for relative imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import MODELS_DIR, ML_CONFIG
from utils.helpers import logger

class TorchLSTM(nn.Module):
    """LSTM implementation using PyTorch with Dropout."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(TorchLSTM, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            if x.shape[1] == (30 * self.input_size):
                 x = x.view(batch_size, 30, self.input_size)
        
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) 
        out = self.fc(out)
        return out

class ModelTrainer:
    """
    Orchestrates training for various model types.
    Handles hyperparameter tuning and model persistence.
    """
    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.models = {}
        self.best_params = {}

    def train_linear_regression(self, X_train, y_train) -> LinearRegression:
        """Baseline Linear Regression."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models["linear_regression"] = model
        return model

    def train_random_forest(self, X_train, y_train) -> RandomForestRegressor:
        """Random Forest with parameter tuning."""
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, random_state=42)
        search.fit(X_train, y_train)
        self.models["random_forest"] = search.best_estimator_
        return search.best_estimator_

    def train_xgboost(self, X_train, y_train) -> XGBRegressor:
        """XGBoost with parameter tuning."""
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1]
        }
        search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, random_state=42)
        search.fit(X_train, y_train)
        self.models["xgboost"] = search.best_estimator_
        return search.best_estimator_

    def train_lightgbm(self, X_train, y_train) -> LGBMRegressor:
        """LightGBM with parameter tuning."""
        model = LGBMRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'num_leaves': [31, 62],
            'learning_rate': [0.01, 0.1]
        }
        search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, random_state=42)
        search.fit(X_train, y_train)
        self.models["lightgbm"] = search.best_estimator_
        return search.best_estimator_

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32) -> TorchLSTM:
        """LSTM training with Early Stopping and validation monitoring."""
        input_size = X_train.shape[2]
        model = TorchLSTM(input_size=input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to Tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).view(-1, 1)
        
        best_val_loss = float('inf')
        patience = 7
        trigger_times = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            if epoch == 0: logger.info(f"LSTM Train Input Shape: {X_train_t.shape}")
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            # Early Stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                # Save best model state
                best_model_state = model.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
        model.load_state_dict(best_model_state)
        self.models["lstm"] = model
        return model

    def save_models(self, features: List[str]):
        """Persist all trained models and features used."""
        # Save features metadata
        metadata_path = MODELS_DIR / f"{self.asset_id}_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({"features": features}, f)
            
        for name, model in self.models.items():
            path = MODELS_DIR / f"{self.asset_id}_{name}.pkl"
            if name == "lstm":
                torch.save(model.state_dict(), path)
            else:
                with open(path, "wb") as f:
                    pickle.dump(model, f)
        logger.info(f"All models for {self.asset_id} saved to {MODELS_DIR}")
