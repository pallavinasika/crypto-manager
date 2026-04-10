import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_models.predictor import CryptoPricePredictor

def generate_mock_data(days=100):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    prices = np.cumsum(np.random.randn(days) * 2) + 500
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "price": prices.astype(float),
        "total_volume": (np.random.rand(days) * 1e9).astype(float),
        "market_cap": (prices * 1e7).astype(float)
    })

predictor = CryptoPricePredictor()
df = generate_mock_data()

models_to_test = [
    ("Gradient Boosting", predictor.train_gradient_boosting),
    ("Random Forest", predictor.train_random_forest),
    ("Decision Tree", predictor.train_decision_tree),
    ("Linear Regression", predictor.train_linear_regression)
]

for name, train_func in models_to_test:
    print(f"\n--- Testing {name} training with tuning ---")
    try:
        result = train_func(df, "test_coin")
        print(f"✅ {name} Success!")
        print(f"Metrics: {result['metrics']}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ {name} Failed: {e}")
