import asyncio
from database.mongo_connection import connect_to_mongo, get_database

async def clean_alerts():
    await connect_to_mongo()
    db = get_database()
    
    # 1. Delete alerts missing coin_id
    result1 = await db["alerts"].delete_many({"coin_id": {"$exists": False}})
    # 2. Delete alerts with empty/null coin_id
    result2 = await db["alerts"].delete_many({"coin_id": {"$in": [None, ""]}})
    # 3. Delete alerts missing threshold
    result3 = await db["alerts"].delete_many({"threshold": {"$exists": False}})
    # 4. Delete alerts with null threshold
    result4 = await db["alerts"].delete_many({"threshold": None})
    
    total_deleted = result1.deleted_count + result2.deleted_count + result3.deleted_count + result4.deleted_count
    print(f"Deleted {total_deleted} malformed alerts.")

if __name__ == "__main__":
    asyncio.run(clean_alerts())
