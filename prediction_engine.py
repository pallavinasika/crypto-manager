import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import torch
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Modular imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_models.data_preprocessing import DataPreprocessor
from ai_models.model_training import TorchLSTM
from ai_models.evaluation import ModelEvaluator
from config.settings import MODELS_DIR

class PredictionEngine:
    """
    High-level engine to manage multi-asset predictions.
    Implements weighted ensembles and stable forecasting.
    """
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.weights = {} # Weights based on validation metrics

    def load_asset_models(self, asset_id: str, model_names: List[str]):
        """Load persistent models for a specific asset."""
        self.models[asset_id] = {}
        
        # Load features metadata
        metadata_path = MODELS_DIR / f"{asset_id}_metadata.pkl"
        features = []
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                features = metadata.get("features", [])
        
        for name in model_names:
            path = MODELS_DIR / f"{asset_id}_{name}.pkl"
            if not path.exists():
                continue
                
            if name == "lstm":
                # Initialize LSTM with the correct input size from metadata
                input_size = len(features) if features else 8
                model = TorchLSTM(input_size=input_size)
                model.load_state_dict(torch.load(path))
                model.eval()
                self.models[asset_id][name] = model
            else:
                with open(path, "rb") as f:
                    self.models[asset_id][name] = pickle.load(f)

    def forecast(self, df: pd.DataFrame, asset_id: str, days_ahead: int = 7) -> Dict:
        """
        Produce a stable multi-model forecast.
        """
        # 1. Clean and engineer features
        clean_df = self.preprocessor.clean_data(df)
        eng_df = self.preprocessor.engineer_features(clean_df)
        
        # 2. Get scaled data (use existing scaler if available)
        scaled_data, features = self.preprocessor.get_scaled_data(eng_df, asset_id, is_training=False)
        
        results = {}
        predictions_all = []
        
        # Determine weights (simplification: equal weights if not pre-calculated)
        model_list = list(self.models.get(asset_id, {}).keys())
        if not model_list:
            return {"error": f"No models found for {asset_id}"}
            
        weights = {m: 1.0/len(model_list) for m in model_list}
        
        # Recursive future prediction
        last_price = float(df["price"].iloc[-1])
        current_data = clean_df.copy()
        
        future_preds = []
        last_date = pd.to_datetime(df["date"].iloc[-1])
        
        for d in range(1, days_ahead + 1):
            day_preds = []
            
            # Recalculate features for the current "future" step
            eng_future = self.preprocessor.engineer_features(current_data)
            scaled_future, _ = self.preprocessor.get_scaled_data(eng_future, asset_id, is_training=False)
            
            for name, model in self.models[asset_id].items():
                if name == "lstm":
                    seq_len = 30
                    if len(scaled_future) >= seq_len:
                        X_seq = scaled_future[-seq_len:].reshape(1, seq_len, -1)
                        with torch.no_grad():
                            X_t = torch.FloatTensor(X_seq)
                            pred = float(model(X_t).item())
                    else:
                        pred = last_price
                else:
                    # Non-LSTM models were trained on flattened sequences (240 features)
                    seq_len = 30
                    if len(scaled_future) >= seq_len:
                        X_flat = scaled_future[-seq_len:].reshape(1, -1)
                        pred = float(model.predict(X_flat)[0])
                    else:
                         # Fallback to single-row if sequence unavailable (shouldn't happen with 200 days data)
                         X_input = scaled_future[-1:].reshape(1, -1)
                         try:
                             pred = float(model.predict(X_input)[0])
                         except:
                             pred = last_price
                
                # Stability: Clipping
                pred = max(pred, last_price * 0.5) # No more than 50% drop per step
                pred = min(pred, last_price * 2.0) # No more than 100% gain per step
                day_preds.append(pred * weights[name])
                
            final_pred = sum(day_preds)
            future_preds.append({
                "date": (last_date + timedelta(days=d)).strftime("%Y-%m-%d"),
                "price": final_pred
            })
            
            # Update current_data for next recursive step
            new_row = pd.DataFrame({
                "date": [last_date + timedelta(days=d)],
                "price": [final_pred]
            })
            current_data = pd.concat([current_data, new_row], ignore_index=True)

        return {
            "asset_id": asset_id,
            "current_price": last_price,
            "forecast": future_preds,
            "confidence": 0.85, # Logic for dynamic confidence here
            "risk_level": "Medium" if 0.4 < 0.85 < 0.8 else "Low" # Placeholder logic
        }
