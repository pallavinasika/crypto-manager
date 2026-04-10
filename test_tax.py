import asyncio
from backend.api.server import get_tax_report, User
from database.mongo_connection import connect_to_mongo
from bson import ObjectId

async def main():
    await connect_to_mongo()
    # Use the mock user Paramesh
    user = User(id=ObjectId("65e9b1e8a1d2c3b4e5f6a7b9"), email="bhupathipramesh2025@gmail.com", name="Paramesh", hashed_password="mock", created_at="2024-01-01")
    try:
        result = await get_tax_report(user)
        print("Tax Report Status:", result.get("status"))
        if "data" in result:
            print("Data keys:", result["data"].keys())
            print("Summary:", result["data"].get("summary"))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
