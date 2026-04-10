import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Simple mock for config/settings if needed
# (Assuming the files are already in place and imports work)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_models.pipeline import AIPredictionPipeline
from utils.helpers import logger

def generate_mock_data(symbol="BTC", days=100):
    """Generate mock historical data for testing."""
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Random walk for price
    price = 50000
    prices = []
    for _ in range(days):
        price = price * (1 + np.random.normal(0, 0.02))
        prices.append(price)
        
    return pd.DataFrame({
        "date": dates,
        "price": prices,
        "total_volume": [1000000] * days,
        "market_cap": [1000000000] * days,
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices]
    })

def main():
    import traceback
    try:
        print("🚀 Verifying Modular AI Prediction System...\n")
        pipeline = AIPredictionPipeline()
        assets = ["bitcoin"] # Testing with just Bitcoin for faster debug
        
        for asset in assets:
            print(f"--- Testing {asset.upper()} ---")
            df = generate_mock_data(asset, days=200)
            
            # 1. Test Training
            print(f"  [1/3] Training cycle...")
            print(f"    - Input Data Shape: {df.shape}")
            result_train = pipeline.run_training_cycle(df, asset)
            if "error" in result_train:
                print(f"    ❌ Training Failed: {result_train['error']}")
                continue
            
            # 2. Test Prediction
            print(f"  [2/3] Prediction engine...")
            print(f"    - Calling get_prediction...")
            result_pred = pipeline.get_prediction(df, asset, days_ahead=7)
            if "error" in result_pred:
                print(f"    ❌ Prediction Failed: {result_pred['error']}")
                continue
                
            print(f"    ✅ Prediction SUCCESS")
            
    except Exception as e:
        with open("tmp/verify_output_2.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"❌ Error encountered, check tmp/verify_output_2.txt")
        raise e

if __name__ == "__main__":
    main()
