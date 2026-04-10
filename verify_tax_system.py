import asyncio
import os
import sys
from datetime import datetime, timedelta
from bson import ObjectId

# Add root and backend to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'backend'))

os.environ["MONGODB_OFFLINE_FILE"] = os.path.join(os.getcwd(), "database", "offline_storage.json")

from database.mongo_connection import get_database, connect_to_mongo, close_mongo_connection
from services.portfolio_manager import PortfolioManager
from services.report_generator import ReportGenerator

PARAMESH_USER_ID = ObjectId("65e9b1e8a1d2c3b4e5f6a7c0")

async def verify_tax():
    await connect_to_mongo()
    db = get_database()
    pm = PortfolioManager()
    rg = ReportGenerator()
    
    print("--- 🧪 Verifying Tax Report System ---")
    
    # 1. Clear existing transactions for test user
    await db["transactions"].delete_many({"user_id": PARAMESH_USER_ID})
    print("🧹 Cleared old transactions.")
    
    # 2. Seed discrete transactions
    # Scenario: 
    # Buy 1 BTC @ 50k
    # Buy 1 BTC @ 60k
    # Sell 1.5 BTC @ 70k -> FIFO: 1 BTC (from 50k) + 0.5 BTC (from 60k)
    # Gain = 1*(70-50) + 0.5*(70-60) = 20k + 5k = 25k realized gain.
    # Remaining: 0.5 BTC @ 60k cost basis.
    
    now = datetime.utcnow()
    # Manual insertion to simulate history
    test_txs = [
        {"user_id": PARAMESH_USER_ID, "coin_id": "bitcoin", "type": "BUY", "quantity": 1.0, "price": 50000.0, "timestamp": now - timedelta(days=10)},
        {"user_id": PARAMESH_USER_ID, "coin_id": "bitcoin", "type": "BUY", "quantity": 1.0, "price": 60000.0, "timestamp": now - timedelta(days=5)},
        {"user_id": PARAMESH_USER_ID, "coin_id": "bitcoin", "type": "SELL", "quantity": 1.5, "price": 70000.0, "timestamp": now - timedelta(days=1)}
    ]
    await db["transactions"].insert_many(test_txs)
    print("✅ Seeded 3 BTC transactions (FIFO test).")
    
    # 3. Add some ETH
    await db["transactions"].insert_one({
        "user_id": PARAMESH_USER_ID, "coin_id": "ethereum", "type": "BUY", "quantity": 10.0, "price": 3000.0, "timestamp": now - timedelta(days=2)
    })
    
    # 4. Fetch transactions and current prices
    all_txs = await db["transactions"].find({"user_id": PARAMESH_USER_ID}).to_list(length=100)
    
    # Mock current prices
    # BTC @ 75k, ETH @ 3500
    market_data = {
        "bitcoin": 75000.0,
        "ethereum": 3500.0
    }
    
    # 5. Generate tax report logic
    report = rg.generate_tax_report(all_txs, market_data)
    
    print("\n--- 📊 Report Summary ---")
    summary = report["summary"]
    print(f"Total Realized Gain:  ${summary['total_realized_gain']:,.2f}")
    print(f"Total Realized Loss:  ${summary['total_realized_loss']:,.2f}")
    print(f"Total Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
    print(f"Total Cost Basis:     ${summary['total_cost_basis']:,.2f}")
    
    # Expected:
    # Realized Gain: 25,000
    # Unrealized BTC: 0.5 * (75k - 60k) = 7,500
    # Unrealized ETH: 10 * (3500 - 3000) = 5,000
    # Total Unrealized: 12,500
    
    if abs(summary['total_realized_gain'] - 25000) < 0.1:
        print("\n✅ FIFO Realized Gain matches expected: $25,000.00")
    else:
        print(f"\n❌ Error: Realized Gain mismatch. Expected 25000, got {summary['total_realized_gain']}")

    if abs(summary['total_unrealized_pnl'] - 12500) < 0.1:
        print("✅ Unrealized P&L matches expected: $12,500.00")
    else:
        print(f"❌ Error: Unrealized P&L mismatch. Expected 12500, got {summary['total_unrealized_pnl']}")
        
    await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(verify_tax())
