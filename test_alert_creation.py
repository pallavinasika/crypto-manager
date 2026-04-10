import asyncio
import os
import sys
from pathlib import Path

# Adjust sys.path to root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.mongo_connection import connect_to_mongo, get_database
from backend.services.alert_system import AlertSystem
from bson import ObjectId

async def test_create_alert():
    print("Connecting to MongoDB/MemoryDB...")
    await connect_to_mongo()
    
    alert_system = AlertSystem()
    user_id = "65e9b1e8a1d2c3b4e5f6a7b9" # Paramesh from offline_storage.json
    coin_id = "bitcoin"
    alert_type = "price_above"
    threshold = 75000.0
    
    print(f"Creating alert for user {user_id}, coin {coin_id}, threshold {threshold}...")
    alert_id = await alert_system.create_price_alert(user_id, coin_id, alert_type, threshold)
    
    if alert_id:
        print(f"✅ Alert created successfully! ID: {alert_id}")
        
        db = get_database()
        alert = await db["alerts"].find_one({"_id": ObjectId(alert_id)})
        print(f"Retrieved alert from DB: {alert}")
    else:
        print("❌ Failed to create alert.")

if __name__ == "__main__":
    asyncio.run(test_create_alert())
