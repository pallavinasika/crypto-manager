"""
AI Prediction Engine - Modular Wrapper
Refactored to delegate to specialized modules: 
data_preprocessing, model_training, prediction_engine, and evaluation.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta

# Modular imports
from ai_models.pipeline import AIPredictionPipeline
from utils.helpers import logger

class CryptoPricePredictor:
    """
    Wrapper for the modular AI Prediction Pipeline.
    Maintains compatibility with existing server API.
    """
    def __init__(self):
        self.pipeline = AIPredictionPipeline()

    def get_ohlcv_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        """Fetch clean OHLCV data using yfinance."""
        symbol_map = {
            "bitcoin": "BTC-USD",
            "ethereum": "ETH-USD",
            "solana": "SOL-USD",
            "cardano": "ADA-USD",
        }
        symbol = symbol_map.get(coin_id, f"{coin_id.upper()}-USD")
        
        try:
            logger.info(f"Fetching OHLCV data for {symbol} from yfinance...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            
            if df.empty:
                return pd.DataFrame()
                
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "price",
                "Volume": "total_volume"
            })
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return pd.DataFrame()

    def _format_prediction(self, prediction: Dict) -> Dict:
        """Standardize prediction format for frontend and CLI."""
        if "forecast" in prediction:
            # Ensure 'predictions' list exists for Recharts
            if "predictions" not in prediction:
                prediction["predictions"] = [{"date": p["date"], "predicted_price": p["price"]} for p in prediction["forecast"]]
            
            # Ensure 'predicted_price' exists (use last forecast point)
            if "predicted_price" not in prediction:
                prediction["predicted_price"] = prediction["forecast"][-1]["price"]
            
            # Add derived metrics
            current = prediction.get("current_price", 0)
            if current > 0:
                change_pct = (prediction["predicted_price"] / current - 1) * 100
                prediction["predicted_change_pct"] = change_pct
                prediction["prediction_direction"] = "Bullish" if change_pct > 2 else ("Bearish" if change_pct < -2 else "Neutral")
            else:
                prediction["predicted_change_pct"] = 0
                prediction["prediction_direction"] = "Neutral"
                
        return prediction

    def check_models_exist(self, asset_id: str) -> bool:
        """Check if models for this asset already exist on disk."""
        model_names = ["linear_regression", "random_forest", "xgboost", "lstm"]
        from config.settings import MODELS_DIR
        for name in model_names:
            path = MODELS_DIR / f"{asset_id}_{name}.pkl"
            if path.exists():
                return True
        return False

    def prepare_lite_prediction(self, df: pd.DataFrame, coin_id: str) -> Dict:
        """Provide a simple statistical prediction when ML models are training."""
        if df.empty or len(df) < 20:
            return {"error": "Insufficient data for lite fallback"}
            
        prices = df["price"].tolist()
        last_price = prices[-1]
        
        # Simple 7-day Moving Average trend
        ma7 = sum(prices[-7:]) / 7
        trend = (last_price / ma7 - 1) * 100
        
        # Simple projection: continue trend with damping
        projected = last_price * (1 + (trend * 0.01))
        
        # Format like a real prediction
        prediction = {
            "coin_id": coin_id,
            "current_price": last_price,
            "predicted_price": projected,
            "predicted_change_pct": trend,
            "prediction_direction": "Bullish" if trend > 0.5 else ("Bearish" if trend < -0.5 else "Neutral"),
            "confidence": 0.3, # Low confidence for lite fallback
            "forecast": [
                {"date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"), "price": last_price * (1 + (trend * 0.002 * i))}
                for i in range(1, 15)
            ],
            "fallback": True
        }
        return self._format_prediction(prediction)

    def predict_future_prices(self, df: pd.DataFrame, coin_id: str, 
                               model_type: str = "random_forest", 
                               days_ahead: int = 7) -> Dict:
        """Compatibility method for single-model prediction."""
        prediction = self.pipeline.get_prediction(df, coin_id, days_ahead)
        if "error" in prediction:
            return self.prepare_lite_prediction(df, coin_id)
        return self._format_prediction(prediction)

    def ensemble_predict(self, df: pd.DataFrame, coin_id: str, days_ahead: int = 7) -> Dict:
        """Delegate to the modular pipeline ensemble with lite fallback."""
        prediction = self.pipeline.get_prediction(df, coin_id, days_ahead)
        if "error" in prediction:
            return self.prepare_lite_prediction(df, coin_id)
        return self._format_prediction(prediction)
