import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from database.mongo_connection import connect_to_mongo, get_database, get_storage_name

async def test_connection():
    url = os.getenv('MONGODB_URL')
    print(f"--- MongoDB Connection Test ---")
    print(f"URL: {url[:30]}..." if url else "URL: MISSING")
    
    try:
        await connect_to_mongo()
        storage = get_storage_name()
        print(f"Final Active Storage: {storage}")
        
        if storage == "MongoDB":
            db = get_database()
            # Test a simple operation
            await db["test_connection"].insert_one({"test": True, "timestamp": "now"})
            print("✅ SUCCESS: Successfully written to MongoDB Atlas!")
        else:
            print("❌ FAILURE: Application is in Offline Mode.")
    except Exception as e:
        print(f"❌ ERROR during test: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
