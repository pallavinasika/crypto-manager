"""
Alert System Module - Handles price alerts and notifications.
"""

import asyncio
from typing import List, Dict
from datetime import datetime
from database.mongo_connection import get_database
from backend.services.notification_service import NotificationService
from utils.helpers import logger

class AlertSystem:
    """
    Monages crypto price alerts and triggers notifications when thresholds are met.
    """
    def __init__(self):
        self.notifier = NotificationService()

    async def create_price_alert(self, user_id, asset, alert_type, price_threshold):
        """
        Create a new price alert in the database.
        """
        try:
            # 1. Validation
            if not asset:
                logger.error("❌ Alert creation failed: Missing asset")
                return None
            
            try:
                price_threshold = float(price_threshold)
            except (ValueError, TypeError):
                logger.error(f"❌ Alert creation failed: Invalid price threshold {price_threshold}")
                return None

            if alert_type not in ["price_above", "price_below"]:
                logger.error(f"❌ Alert creation failed: Invalid alert type {alert_type}")
                return None

            db = get_database()
            
            # 2. Prepare document matching user's requested schema
            alert_doc = {
                "userId": user_id,
                "asset": asset,
                "alertType": alert_type,
                "priceThreshold": price_threshold,
                "createdAt": datetime.utcnow(),
                "status": "active" # status (active / triggered)
            }
            
            # 3. Insert and return ID
            result = await db["alerts"].insert_one(alert_doc)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error creating price alert in AlertSystem: {e}")
            return None

    async def check_alerts(self, market_data: List[Dict]):
        """
        Check all active alerts against current market data.
        """
        triggered = []
        db = get_database()
        
        # 1. Map market data for O(1) lookup: { "bitcoin": 65000, ... }
        price_map = {}
        for coin in market_data:
            cid = coin.get("coin_id") or coin.get("id")
            if cid:
                price_map[cid] = coin.get("price", 0)

        try:
            # 2. Only fetch active, untriggered alerts using NEW SCHEMA
            cursor = db["alerts"].find({"status": "active"})
            
            async for alert in cursor:
                asset = alert.get("asset")
                if not asset or asset not in price_map:
                    continue
                    
                current_price = price_map[asset]
                threshold = alert.get("priceThreshold", 0)
                alert_type = alert.get("alertType")
                
                # 3. Logic check
                is_triggered = False
                if alert_type == "price_above" and current_price >= threshold:
                    is_triggered = True
                elif alert_type == "price_below" and current_price <= threshold:
                    is_triggered = True

                if is_triggered:
                    # Update DB immediately to prevent double-triggering
                    await db["alerts"].update_one(
                        {"_id": alert["_id"]},
                        {"$set": {"status": "triggered", "triggered_at": datetime.utcnow()}}
                    )
                    
                    # 4. Prepare notification message
                    msg = f"🚀 {asset.upper()} Alert: {current_price} reached {threshold}!"
                    
                    # Fire and forget notifications
                    asyncio.create_task(self.notifier.send_discord_webhook("Alert", msg))
                    asyncio.create_task(self.notifier.send_telegram_message(msg))
                    
                    triggered.append({
                        "alert_id": str(alert["_id"]), 
                        "user_id": str(alert.get("userId")),
                        "message": msg
                    })

            return triggered
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []