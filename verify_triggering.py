import asyncio
import os
import sys
from pathlib import Path

# Adjust sys.path to root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.mongo_connection import connect_to_mongo, get_database
from backend.services.alert_system import AlertSystem
from bson import ObjectId

async def test_triggering():
    print("Connecting to MongoDB/MemoryDB...")
    await connect_to_mongo()
    
    alert_system = AlertSystem()
    user_id = "65e9b1e8a1d2c3b4e5f6a7b9" # Paramesh
    coin_id = "bitcoin"
    alert_type = "price_above"
    threshold = 70000.0
    
    print(f"Creating alert for {coin_id} at {threshold}...")
    alert_id = await alert_system.create_price_alert(user_id, coin_id, alert_type, threshold)
    
    if alert_id:
        print(f"✅ Alert created: {alert_id}")
        
        # Mock market data from CoinGecko (using 'id' instead of 'coin_id')
        market_data = [
            {"id": "bitcoin", "current_price": 71000.0, "price": 71000.0},
            {"id": "ethereum", "current_price": 3500.0, "price": 3500.0}
        ]
        
        print("Checking alerts with mock data...")
        triggered = await alert_system.check_alerts(market_data)
        
        if triggered:
            print(f"✅ Alert triggered correctly! {triggered}")
        else:
            print("❌ Alert was NOT triggered.")
    else:
        print("❌ Failed to create alert.")

if __name__ == "__main__":
    asyncio.run(test_triggering())
