"""
Portfolio Manager Module - Manages cryptocurrency portfolios using MongoDB.
Handles CRUD operations, P&L calculations, and allocation tracking.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from bson import ObjectId
from database.mongo_connection import get_database
from utils.helpers import logger

class PortfolioManager:
    """
    Manages user cryptocurrency portfolios.
    Provides CRUD operations, value tracking, P&L analysis,
    and allocation calculations.
    """

    def __init__(self):
        pass

    async def create_portfolio(self, user_id: str, name: str = "My Portfolio", description: str = "") -> str:
        """Create a new portfolio. Returns portfolio ID."""
        db = get_database()
        try:
            portfolio = {
                "user_id": ObjectId(user_id),
                "name": name,
                "description": description,
                "assets": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            result = await db["portfolios"].insert_one(portfolio)
            logger.info(f"Created portfolio '{name}' for user {user_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            return None

    async def get_portfolio(self, portfolio_id: str) -> Optional[Dict]:
        """Get portfolio details including all items."""
        db = get_database()
        try:
            portfolio = await db["portfolios"].find_one({"_id": ObjectId(portfolio_id)})
            if not portfolio:
                return None

            items = []
            total_value = 0
            total_cost = 0

            for asset in portfolio.get("assets", []):
                # Fetch current price from market_data collection (latest)
                market_data = await db["market_data"].find_one(
                    {"coin_id": asset["coin_id"]},
                    sort=[("timestamp", -1)]
                )
                
                # Fetch metadata
                crypto = await db["cryptocurrencies"].find_one({"coin_id": asset["coin_id"]})
                
                current_price = market_data["price"] if market_data else 0
                current_value = asset["quantity"] * current_price
                cost_basis = asset["quantity"] * asset["purchase_price"]
                profit_loss = current_value - cost_basis
                profit_loss_pct = (profit_loss / cost_basis * 100) if cost_basis > 0 else 0

                total_value += current_value
                total_cost += cost_basis

                items.append({
                    "coin_id": asset["coin_id"],
                    "name": crypto["name"] if crypto else asset["coin_id"].capitalize(),
                    "symbol": crypto["symbol"] if crypto else "",
                    "quantity": asset["quantity"],
                    "purchase_price": asset["purchase_price"],
                    "current_price": current_price,
                    "current_value": current_value,
                    "cost_basis": cost_basis,
                    "profit_loss": profit_loss,
                    "profit_loss_pct": profit_loss_pct,
                    "purchase_date": asset.get("purchase_date"),
                })

            # Calculate allocation percentages
            for item in items:
                item["allocation_pct"] = (
                    (item["current_value"] / total_value * 100) if total_value > 0 else 0
                )

            total_pl = total_value - total_cost
            total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

            return {
                "id": str(portfolio["_id"]),
                "name": portfolio["name"],
                "description": portfolio.get("description"),
                "total_value": total_value,
                "total_cost": total_cost,
                "total_pl": total_pl,
                "total_pl_pct": total_pl_pct,
                "assets": items,
                "num_assets": len(items)
            }
        except Exception as e:
            logger.error(f"Error fetching portfolio {portfolio_id}: {e}")
            return None

    async def _record_transaction(self, user_id: ObjectId, coin_id: str, type: str, quantity: float, price: float, portfolio_id: ObjectId):
        """Hidden helper to record individual transaction history for tax purposes."""
        db = get_database()
        try:
            tx = {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "coin_id": coin_id,
                "type": type.upper(), # BUY, SELL, TRANSFER, STAKING
                "quantity": abs(quantity),
                "price": price,
                "total_value": abs(quantity) * price,
                "timestamp": datetime.utcnow()
            }
            await db["transactions"].insert_one(tx)
            logger.info(f"Recorded {type} transaction for {coin_id}")
        except Exception as e:
            logger.error(f"Failed to record transaction: {e}")

    async def add_asset(self, portfolio_id: str, coin_id: str, quantity: float, purchase_price: float) -> bool:
        """Add an asset to a portfolio, averaging into existing if found (DCA)."""
        db = get_database()
        try:
            # 1. Fetch current portfolio
            portfolio = await db["portfolios"].find_one({"_id": ObjectId(portfolio_id)})
            if not portfolio:
                logger.error(f"Portfolio {portfolio_id} not found")
                return False
            
            user_id = portfolio["user_id"]
            
            # 2. Record the discrete transaction for tax history
            await self._record_transaction(user_id, coin_id, "BUY", quantity, purchase_price, ObjectId(portfolio_id))
            
            assets = portfolio.get("assets", [])
            existing_idx = next((i for i, a in enumerate(assets) if a.get("coin_id") == coin_id), None)
            
            if existing_idx is not None:
                # 2. Average with existing asset (DCA logic)
                existing = assets[existing_idx]
                old_qty = existing.get("quantity", 0)
                old_price = existing.get("purchase_price", 0)
                
                new_total_qty = old_qty + quantity
                if new_total_qty > 0:
                    # Weighted average: (Q1*P1 + Q2*P2) / (Q1+Q2)
                    new_avg_price = ((old_qty * old_price) + (quantity * purchase_price)) / new_total_qty
                else:
                    new_avg_price = purchase_price

                # Replace item in list (Atomic update for the whole array for compatibility)
                assets[existing_idx] = {
                    "coin_id": coin_id,
                    "quantity": new_total_qty,
                    "purchase_price": new_avg_price,
                    "purchase_date": datetime.utcnow()
                }
                
                await db["portfolios"].update_one(
                    {"_id": ObjectId(portfolio_id)},
                    {"$set": {"assets": assets, "updated_at": datetime.utcnow()}}
                )
                logger.info(f"Averaged {coin_id} in portfolio {portfolio_id}: {new_total_qty} @ {new_avg_price}")
            else:
                # 3. Add as new asset
                asset = {
                    "coin_id": coin_id,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "purchase_date": datetime.utcnow()
                }
                await db["portfolios"].update_one(
                    {"_id": ObjectId(portfolio_id)},
                    {"$push": {"assets": asset}, "$set": {"updated_at": datetime.utcnow()}}
                )
                logger.info(f"Added new asset {coin_id} to portfolio {portfolio_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding asset to portfolio {portfolio_id}: {e}")
            return False

    async def remove_asset(self, portfolio_id: str, coin_id: str) -> bool:
        """Remove all holdings of a specific coin, recording a full SELL transaction."""
        db = get_database()
        try:
            portfolio = await db["portfolios"].find_one({"_id": ObjectId(portfolio_id)})
            if not portfolio: return False
            
            # Find the asset to get quantity
            asset = next((a for a in portfolio.get("assets", []) if a["coin_id"] == coin_id), None)
            if asset:
                # Fetch current market price for the sell record
                market_data = await db["market_data"].find_one({"coin_id": coin_id}, sort=[("timestamp", -1)])
                current_price = market_data["price"] if market_data else asset["purchase_price"]
                
                await self._record_transaction(
                    portfolio["user_id"], 
                    coin_id, "SELL", 
                    asset["quantity"], 
                    current_price, 
                    ObjectId(portfolio_id)
                )

            await db["portfolios"].update_one(
                {"_id": ObjectId(portfolio_id)},
                {"$pull": {"assets": {"coin_id": coin_id}}, "$set": {"updated_at": datetime.utcnow()}}
            )
            return True
        except Exception as e:
            logger.error(f"Error removing asset {coin_id} from portfolio {portfolio_id}: {e}")
            return False

    async def list_portfolios(self, user_id: str) -> List[Dict]:
        """List all portfolios for a user."""
        db = get_database()
        try:
            cursor = db["portfolios"].find({"user_id": ObjectId(user_id)})
            portfolios = []
            async for p in cursor:
                portfolios.append(await self.get_portfolio(str(p["_id"])))
            return portfolios
        except Exception as e:
            logger.error(f"Error listing portfolios for user {user_id}: {e}")
            return []
