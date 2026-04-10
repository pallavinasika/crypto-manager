import asyncio
from backend.services.data_collector import CryptoDataCollector
from database.mongo_connection import connect_to_mongo

async def main():
    await connect_to_mongo()
    collector = CryptoDataCollector()
    try:
        data = await collector.get_latest_market_data(5)
        print("Market Data Length:", len(data))
        if data:
            print("First item keys:", data[0].keys())
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(main())
