import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

# Fix for relative imports if needed
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from utils.helpers import logger

class DataPreprocessor:
    """
    Handles all data cleaning, feature engineering, and scaling.
    Supports multi-asset isolated processing.
    """
    def __init__(self):
        self.scalers = {}  # One scaler per asset
        self.feature_cols = [
            "price", "sma_7", "sma_14", "sma_30", "rsi", 
            "volatility_7", "daily_return", "day_of_week"
        ]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, handle missing values and outliers."""
        if df.empty:
            return df
            
        data = df.copy()
        
        # 1. Handle missing price values
        if "price" in data.columns:
            data = data.dropna(subset=["price"])
            
        # 2. Sort by date
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date")
            
        # 3. Remove duplicates
        data = data.drop_duplicates(subset=["date"] if "date" in data.columns else None)
        
        # 4. Outlier removal (Price specifically)
        # Using 3-sigma rule for daily returns
        if len(data) > 20:
            returns = data["price"].pct_change().dropna()
            mean_ret = returns.mean()
            std_ret = returns.std()
            # We don't drop rows but we can clip extreme returns mapping back to prices
            # (Simplification: clipping prices to 5x rolling median)
            median_price = data["price"].rolling(window=20, center=True).median().fillna(data["price"].median())
            data["price"] = data["price"].clip(lower=median_price * 0.1, upper=median_price * 10)
            
        return data

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators as requested."""
        data = df.copy()
        
        # Moving Averages
        data["sma_7"] = data["price"].rolling(window=7).mean()
        data["sma_14"] = data["price"].rolling(window=14).mean()
        data["sma_30"] = data["price"].rolling(window=30).mean()
        
        # Daily Return
        data["daily_return"] = data["price"].pct_change().fillna(0)
        
        # Volatility (Rolling Std Dev of returns)
        data["volatility_7"] = data["daily_return"].rolling(window=7).std().fillna(0)
        
        # RSI (Relative Strength Index)
        delta = data["price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        data["rsi"] = 100 - (100 / (1 + rs))
        
        # Time Features
        if "date" in data.columns:
            data["day_of_week"] = data["date"].dt.dayofweek
            
        return data.ffill().fillna(0)

    def get_scaled_data(self, df: pd.DataFrame, asset_id: str, is_training: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Scale features using asset-specific scalers."""
        features = [col for col in self.feature_cols if col in df.columns]
        data_to_scale = df[features].values
        
        if is_training or asset_id not in self.scalers:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_to_scale)
            self.scalers[asset_id] = scaler
        else:
            scaler = self.scalers[asset_id]
            scaled_data = scaler.transform(data_to_scale)
            
        return scaled_data, features

    def prepare_lstm_sequences(self, data: np.ndarray, seq_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flat data to (samples, timesteps, features)."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            # Assuming price is the first column in feature_cols
            y.append(data[i + seq_length, 0])
            
        return np.array(X), np.array(y)

    def split_data(self, X: np.ndarray, y: np.ndarray, train_size=0.8, val_size=0.1) -> Tuple:
        """Chronological split: 80% Train, 10% Val, 10% Test."""
        n = len(X)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
