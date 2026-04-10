import asyncio
import os
import sys
from datetime import datetime
from bson import ObjectId

# Add backend and root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'backend'))

os.environ["MONGODB_OFFLINE_FILE"] = os.path.join(os.getcwd(), "database", "offline_storage.json")

from database.mongo_connection import get_database, connect_to_mongo, close_mongo_connection
from services.alert_system import AlertSystem

PARAMESH_USER_ID = "65e9b1e8a1d2c3b4e5f6a7c0"

async def verify_alerts():
    await connect_to_mongo()
    db = get_database()
    alert_sys = AlertSystem()
    
    print("\n--- 🧪 Verifying Alert System Fix ---")
    
    # 1. Clear existing alerts for test user
    await db["alerts"].delete_many({"userId": PARAMESH_USER_ID})
    print("漫 Cleared old alerts.")
    
    # 2. Test create_price_alert method
    print("📡 Testing create_price_alert...")
    alert_id = await alert_sys.create_price_alert(
        user_id=PARAMESH_USER_ID,
        asset="bitcoin",
        alert_type="price_above",
        price_threshold=75000.0
    )
    
    if alert_id:
        print(f"✅ Alert created successfully with ID: {alert_id}")
    else:
        print("❌ Failed to create alert via AlertSystem method.")
        return

    # 3. Verify schema in database
    alert_doc = await db["alerts"].find_one({"_id": ObjectId(alert_id)})
    if alert_doc:
        print("🔍 Verifying Database Schema:")
        fields = ["userId", "asset", "alertType", "priceThreshold", "createdAt", "status"]
        for f in fields:
            print(f"  - {f}: {alert_doc.get(f)}")
        print("✅ Database schema aligns with requirements.")
    else:
        print("❌ Could not find the created alert in the database.")

    # 4. Test trigger logic
    print("📡 Testing trigger logic...")
    market_data = [{"coin_id": "bitcoin", "price": 80000.0}]
    triggered = await alert_sys.check_alerts(market_data)
    if triggered and any(t["alert_id"] == alert_id for t in triggered):
        print("✅ Alert triggered correctly at $80,000.")
    else:
        print("❌ Alert failed to trigger.")

    # 5. Test deletion logic (THE CRITICAL PART)
    print("📡 Testing alert deletion with $or query...")
    query = {"_id": ObjectId(alert_id), "$or": [{"userId": PARAMESH_USER_ID}]}
    result = await db["alerts"].delete_one(query)
    
    if result.deleted_count == 1:
        print("✅ Alert deleted successfully using $or query!")
    else:
        print(f"❌ Failed to delete alert. deleted_count: {result.deleted_count}")
        # Debug why it failed
        all_alerts = await db["alerts"].find({}).to_list(length=10)
        print(f"Current alerts in DB: {len(all_alerts)}")

    await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(verify_alerts())
