"""
Backtesting Engine - Simulates portfolio performance on historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from utils.helpers import logger

class BacktestingEngine:
    """
    Simulates a trading strategy or portfolio on historical market data.
    """
    def __init__(self, data_collector):
        self.collector = data_collector

    async def run_backtest(self, coin_id: str, initial_capital: float, days: int = 365, strategy: str = "buy_and_hold"):
        """
        Run a backtest on a single asset.
        """
        df = await self.collector.get_price_history_from_db(coin_id, days=days)
        if df.empty:
            return {"error": f"No historical data found for {coin_id}"}
            
        df = df.sort_values("timestamp")
        
        if strategy == "buy_and_hold":
            return self._buy_and_hold(df, initial_capital)
        elif strategy == "moving_average_crossover":
            return self._ma_crossover(df, initial_capital)
        else:
            return {"error": f"Strategy {strategy} not implemented"}

    def _buy_and_hold(self, df: pd.DataFrame, capital: float) -> Dict:
        """Simple buy and hold strategy simulation."""
        buy_price = df["price"].iloc[0]
        units = capital / buy_price
        
        portfolio_values = []
        for index, row in df.iterrows():
            portfolio_values.append({
                "date": row["timestamp"].strftime("%Y-%m-%d"),
                "value": units * row["price"],
                "price": row["price"]
            })
            
        final_value = units * df["price"].iloc[-1]
        pnl = final_value - capital
        pnl_pct = (pnl / capital) * 100
        
        return {
            "strategy": "Buy and Hold",
            "initial_capital": capital,
            "final_value": round(final_value, 2),
            "total_pnl": round(pnl, 2),
            "pnl_percentage": round(pnl_pct, 2),
            "history": portfolio_values
        }

    def _ma_crossover(self, df: pd.DataFrame, capital: float, short_window=10, long_window=30) -> Dict:
        """SMA Crossover strategy (10-day vs 30-day)."""
        df["sma_short"] = df["price"].rolling(window=short_window).mean()
        df["sma_long"] = df["price"].rolling(window=long_window).mean()
        
        cash = capital
        units = 0
        in_position = False
        history = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            price = row["price"]
            sma_s = row["sma_short"]
            sma_l = row["sma_long"]
            
            # Skip until we have enough data for SMAs
            if pd.isna(sma_l):
                history.append({"date": row["timestamp"].strftime("%Y-%m-%d"), "value": cash, "action": "wait"})
                continue
                
            # Signal Detection
            if not in_position and sma_s > sma_l:
                # Buy
                units = cash / price
                cash = 0
                in_position = True
                action = "buy"
            elif in_position and sma_s < sma_l:
                # Sell
                cash = units * price
                units = 0
                in_position = False
                action = "sell"
            else:
                action = "hold"
                
            current_value = cash + (units * price)
            history.append({
                "date": row["timestamp"].strftime("%Y-%m-%d"),
                "value": round(current_value, 2),
                "action": action,
                "price": price
            })
            
        final_value = cash + (units * df["price"].iloc[-1])
        return {
            "strategy": "SMA Crossover",
            "initial_capital": capital,
            "final_value": round(final_value, 2),
            "total_pnl": round(final_value - capital, 2),
            "pnl_percentage": round(((final_value / capital) - 1) * 100, 2),
            "history": history
        }
