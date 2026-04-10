import asyncio
import os
import sys
from pathlib import Path

# Adjust sys.path to root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.mongo_connection import connect_to_mongo, close_mongo_connection
from backend.services.portfolio_manager import PortfolioManager
from bson import ObjectId

async def test_dca_averaging():
    print("\n--- Testing Portfolio DCA Averaging ---")
    
    # Initialize
    await connect_to_mongo()
    pm = PortfolioManager()
    
    # Create a dummy user and portfolio
    user_id = str(ObjectId())
    portfolio_id = await pm.create_portfolio(user_id, "DCA Test Portfolio")
    
    if not portfolio_id:
        print("FAIL: Could not create portfolio")
        await close_mongo_connection()
        return

    try:
        # 1. Add first batch: 1 BTC @ $50,000
        print("Adding 1 BTC @ $50,000...")
        await pm.add_asset(portfolio_id, "bitcoin", 1.0, 50000.0)
        
        # 2. Add second batch: 1 BTC @ $60,000
        print("Adding 1 BTC @ $60,000...")
        await pm.add_asset(portfolio_id, "bitcoin", 1.0, 60000.0)
        
        # 3. Verify
        portfolio = await pm.get_portfolio(portfolio_id)
        assets = portfolio["assets"]
        
        print(f"\nPortfolio has {len(assets)} unique assets.")
        
        if len(assets) != 1:
            print(f"FAIL: Expected 1 asset entry, found {len(assets)}")
        else:
            asset = assets[0]
            qty = asset["quantity"]
            price = asset["purchase_price"]
            print(f"Asset: {asset['coin_id']}")
            print(f"Quantity: {qty} (Expected: 2.0)")
            print(f"Avg Price: ${price:,.2f} (Expected: $55,000.00)")
            
            if qty == 2.0 and abs(price - 55000.0) < 0.01:
                print("\nSUCCESS: DCA Averaging is working correctly!")
            else:
                print("\nFAIL: Values do not match expected results.")
                
    finally:
        # Cleanup (optional, but good practice)
        # await db["portfolios"].delete_one({"_id": ObjectId(portfolio_id)})
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(test_dca_averaging())
