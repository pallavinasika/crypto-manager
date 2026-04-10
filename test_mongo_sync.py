import os
from pymongo import MongoClient
from dotenv import load_dotenv

def test_sync_connection():
    load_dotenv()
    url = os.getenv('MONGODB_URL')
    print(f"Testing SYNC connection to: {url[:30]}...")
    
    try:
        client = MongoClient(url, serverSelectionTimeoutMS=5000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("✅ SYNC SUCCESS: Connection established!")
        
        db = client.get_database()
        print(f"Connected to database: {db.name}")
        
        # Test write
        res = db["test_sync"].insert_one({"test": True})
        print(f"✅ SYNC SUCCESS: Inserted ID {res.inserted_id}")
        
    except Exception as e:
        print(f"❌ SYNC FAILURE: {e}")

if __name__ == "__main__":
    test_sync_connection()
