import sys
sys.path.insert(0, ".")
import traceback
import pandas as pd
import numpy as np
from ai_models.predictor import CryptoPricePredictor

def run_tests():
    print("Generating mock dataset...")
    df = pd.DataFrame({
        'price': np.random.rand(400) * 100 + 100,
        'total_volume': np.random.rand(400) * 1e6
    })
    df['date'] = pd.date_range('2023-01-01', periods=400)
    
    p = CryptoPricePredictor()
    
    print("\n--- Testing Linear Regression ---")
    try:
        p.train_linear_regression(df.copy(), 'bitcoin')
        print("SUCCESS")
    except Exception as e:
        print("FAILED")
        traceback.print_exc()
        
    print("\n--- Testing Decision Tree ---")
    try:
        p.train_decision_tree(df.copy(), 'bitcoin')
        print("SUCCESS")
    except Exception as e:
        print("FAILED")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
