import asyncio
import os
import sys
import json
from pathlib import Path

# Adjust sys.path to root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.mongo_connection import connect_to_mongo, get_database, MOCK_DATA_FILE, json_decoder
from backend.services.alert_system import AlertSystem
from bson import ObjectId

async def test_persistence():
    print("Connecting to MongoDB/MemoryDB...")
    await connect_to_mongo()
    
    alert_system = AlertSystem()
    user_id = "65e9b1e8a1d2c3b4e5f6a7b9" # Paramesh
    coin_id = "ethereum"
    alert_type = "price_below"
    threshold = 3000.0
    
    print(f"Creating alert for eth at {threshold}...")
    alert_id = await alert_system.create_price_alert(user_id, coin_id, alert_type, threshold)
    
    if alert_id:
        print(f"✅ Alert created in memory: {alert_id}")
        
        # Manually flush for test consistency if using mock DB
        db = get_database()
        if hasattr(db, "flush_to_file"):
            print("Force flushing to offline storage...")
            await db.flush_to_file()
        else:
            print("Waiting for debounced save (1 second)...")
            await asyncio.sleep(1.5)
        
        if os.path.exists(MOCK_DATA_FILE):
            print(f"Checking {MOCK_DATA_FILE}...")
            with open(MOCK_DATA_FILE, "r") as f:
                data = json.load(f, object_hook=json_decoder)
                alerts = data.get("alerts", [])
                print(f"Found {len(alerts)} alerts in file.")
                for a in alerts:
                    print(f" - Alert for {a.get('coin_id')} (user: {a.get('user_id')})")
        else:
            print(f"❌ {MOCK_DATA_FILE} does not exist!")
    else:
        print("❌ Failed to create alert.")

if __name__ == "__main__":
    asyncio.run(test_persistence())
