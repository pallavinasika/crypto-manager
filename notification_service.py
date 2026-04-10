"""
Notification Service Module - Handles multi-channel notifications (Push, Webhooks, Telegram).
"""

import httpx
import asyncio
from typing import Dict, List, Optional
from config.settings import (
    FIREBASE_CREDENTIALS_PATH, ENABLE_PUSH_NOTIFICATIONS, WEBHOOK_CONFIG
)
from utils.helpers import logger

# Initialize Firebase Admin if enabled
firebase_app = None
if ENABLE_PUSH_NOTIFICATIONS and FIREBASE_CREDENTIALS_PATH:
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized for push notifications")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")

class NotificationService:
    """
    Manages sending notifications across different platforms.
    """
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def send_push_notification(self, token: str, title: str, body: str, data: Dict = None):
        """Send a push notification via Firebase Cloud Messaging."""
        if not ENABLE_PUSH_NOTIFICATIONS or not firebase_app:
            logger.warning("Push notifications disabled or Firebase not initialized")
            return False
            
        try:
            from firebase_admin import messaging
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                token=token,
            )
            response = messaging.send(message)
            logger.info(f"Successfully sent push notification: {response}")
            return True
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False

    async def send_discord_webhook(self, title: str, message: str, color: int = 0x00ff00):
        """Send an alert to Discord via webhook."""
        webhook_url = WEBHOOK_CONFIG.get("discord_url")
        if not webhook_url:
            return False
            
        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": color,
                "footer": {"text": "AI Crypto Intelligence Alert"}
            }]
        }
        
        try:
            response = await self.client.post(webhook_url, json=payload)
            return response.is_success
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            return False

    async def send_telegram_message(self, message: str):
        """Send a message to a Telegram chat."""
        token = WEBHOOK_CONFIG.get("telegram_token")
        chat_id = WEBHOOK_CONFIG.get("telegram_chat_id")
        
        if not token or not chat_id:
            return False
            
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        
        try:
            response = await self.client.post(url, json=payload)
            return response.is_success
        except Exception as e:
            logger.error(f"Telegram message error: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
