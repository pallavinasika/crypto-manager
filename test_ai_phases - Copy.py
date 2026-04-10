import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import asyncio
from ai_models.predictor import CryptoPricePredictor
from backend.services.data_collector import CryptoDataCollector

async def test_ai_upgrades():
    print("🚀 Testing AI Intelligence Upgrades (Phase 2)")
    
    predictor = CryptoPricePredictor()
    collector = CryptoDataCollector()
    
    # 1. Test Sentiment Analysis integration
    print("\n--- Testing Sentiment Fetching ---")
    sentiment = await collector.fetch_news_sentiment("bitcoin")
    print(f"Bitcoin Sentiment: {sentiment}")
    
    # 2. Test Prediction with Sentiment Features
    print("\n--- Testing Prediction with Sentiment ---")
    # Generate more mock data (200 rows) to ensure indicators have enough room
    df = pd.DataFrame({
        "price": [20000 + i*100 for i in range(200)],
        "total_volume": [1000000] * 200,
        "sentiment_score": [0.1] * 200,
        "date": pd.date_range(start="2024-01-01", periods=200).strftime("%Y-%m-%d")
    })
    
    result = predictor.ensemble_predict(df, "bitcoin", days_ahead=7)
    if "error" in result:
        print(f"❌ Ensemble Prediction failed: {result['error']}")
    else:
        print(f"✅ Ensemble Prediction (Final): {result.get('predicted_price_final', 0):.2f}")
        print(f"Direction: {result.get('prediction_direction', 'N/A')}")
    
    # 3. Test Auto-retraining
    print("\n--- Testing Auto-retraining logic ---")
    needs_retrain = predictor.retrain_if_needed("bitcoin", df, force=True)
    print(f"Forced Retraining Triggered: {needs_retrain}")

if __name__ == "__main__":
    asyncio.run(test_ai_upgrades())
