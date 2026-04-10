import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8001"

def test_prediction_endpoint():
    print("\n--- Testing AI Insights Prediction Endpoint ---")
    
    coin_id = "bitcoin"
    # Testing multiple models to ensure they all work and return predicted_price
    models = ["random_forest", "linear_regression", "ensemble"]
    
    for model in models:
        try:
            url = f"{BASE_URL}/api/predict/{coin_id}?days=30&model={model}"
            print(f"Testing Model: {model} | URL: {url}")
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 401:
                print("Note: Received 401 (Auth required). Endpoint exists and auth is working.")
                continue
                
            if response.status_code != 200:
                print(f"FAIL: Received status {response.status_code}")
                print(f"Response: {response.text}")
                continue
                
            data = response.json()
            if data["status"] == "success":
                prediction_data = data["data"]
                if "predicted_price" in prediction_data:
                    print(f"SUCCESS: 'predicted_price' found: ${prediction_data['predicted_price']:,.2f}")
                else:
                    print(f"FAIL: 'predicted_price' MISSING from response!")
                    print(f"Keys found: {list(prediction_data.keys())}")
            else:
                print(f"FAIL: Status was {data['status']}")
                
        except Exception as e:
            print(f"ERROR connecting: {e}")

if __name__ == "__main__":
    test_prediction_endpoint()
