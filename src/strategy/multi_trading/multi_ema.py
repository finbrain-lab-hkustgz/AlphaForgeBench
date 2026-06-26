"""
Multi EMA Strategy

A multi-asset trading strategy using Exponential Moving Average (EMA) crossover.
Each symbol is traded independently based on its own EMA signals.
"""

from typing import Dict, Any
from pandas import DataFrame
from pydantic import Field
import pandas as pd

from src.strategy.types import Strategy


class MultiEMA(Strategy):
    """Multi-asset EMA crossover strategy.

    For each symbol independently:
    - Buy signal: short EMA crosses above long EMA
    - Sell signal: short EMA crosses below long EMA
    - Position size: configurable weight per symbol

    Supports flexible position allocation:
    - Can allocate full capital (position sum = 100%)
    - Can keep cash reserve (position sum < 100%)
    - Can stay completely in cash (all signals = 0)

    Strategy Output Format:
        {
            "AAPL": {"signal": 1, "position": 0.2},   # Buy 20%
            "GOOGL": {"signal": -1, "position": 0.5},  # Sell 50% of holdings
            "MSFT": {"signal": 0, "position": 0.0},    # Hold
        }
    """

    name: str = Field(default="multi_ema", description="Strategy name")
    description: str = Field(
        default="Multi-asset EMA crossover strategy - trades each symbol independently",
        description="Strategy description"
    )
    factor_names: list = Field(default=[], description="List of factor names used")

    # Strategy parameters
    short_window: int = Field(default=10, description="Short EMA window")
    long_window: int = Field(default=30, description="Long EMA window")
    position_size: float = Field(default=0.2, description="Position size per symbol (0.0 to 1.0)")

    async def __call__(self, data: Dict[str, DataFrame]) -> Dict[str, Dict[str, float]]:
        """Generate trading signals for all symbols based on EMA crossover.

        Args:
            data: Dict mapping symbol to DataFrame (price + features)
                 Each DataFrame has columns: open, high, low, close, volume, etc.

        Returns:
            Dict mapping symbol to {"signal": int, "position": float}
            - signal: 1 (BUY), 0 (HOLD), -1 (SELL)
            - position: ratio of available capital/position
        """
        signals = {}

        for symbol, df in data.items():
            try:
                # Check if we have enough data
                if len(df) < self.long_window:
                    # Not enough data, hold
                    signals[symbol] = {"signal": 0, "position": 0.0}
                    continue

                # Calculate EMAs
                close_prices = df['close']
                short_ema = close_prices.ewm(span=self.short_window, adjust=False).mean()
                long_ema = close_prices.ewm(span=self.long_window, adjust=False).mean()

                # Get current and previous values
                short_ema_curr = short_ema.iloc[-1]
                long_ema_curr = long_ema.iloc[-1]
                short_ema_prev = short_ema.iloc[-2] if len(short_ema) > 1 else short_ema_curr
                long_ema_prev = long_ema.iloc[-2] if len(long_ema) > 1 else long_ema_curr

                # Generate signal based on crossover
                if short_ema_prev <= long_ema_prev and short_ema_curr > long_ema_curr:
                    # Bullish crossover - buy signal
                    signals[symbol] = {"signal": 1, "position": self.position_size}
                elif short_ema_prev >= long_ema_prev and short_ema_curr < long_ema_curr:
                    # Bearish crossover - sell signal
                    signals[symbol] = {"signal": -1, "position": 1.0}  # Sell all holdings
                else:
                    # No crossover - hold
                    signals[symbol] = {"signal": 0, "position": 0.0}

            except Exception as e:
                # On error, default to hold
                signals[symbol] = {"signal": 0, "position": 0.0}

        return signals
