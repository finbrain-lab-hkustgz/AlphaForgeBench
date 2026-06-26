"""
Multi Buy and Hold Strategy

A simple multi-asset buy-and-hold strategy that buys all symbols equally
at the beginning and holds until the end.
"""

from typing import Dict, Any
from pandas import DataFrame
from pydantic import Field

from src.strategy.types import Strategy


class MultiBuyAndHold(Strategy):
    """Multi-asset buy and hold strategy.

    This strategy equally divides capital across all symbols and buys
    them all at the first step, then holds until the end.

    Strategy Output Format:
        {
            "AAPL": {"signal": 1, "position": 0.33},
            "GOOGL": {"signal": 1, "position": 0.33},
            "MSFT": {"signal": 1, "position": 0.34},
        }
    """

    name: str = Field(default="multi_buy_and_hold", description="Strategy name")
    description: str = Field(
        default="Multi-asset buy and hold strategy - equally divides capital across all symbols",
        description="Strategy description"
    )
    factor_names: list = Field(default=[], description="List of factor names used")
    has_bought: bool = Field(default=False, description="Track if initial buy has been executed")

    async def __call__(self, data: Dict[str, DataFrame]) -> Dict[str, Dict[str, float]]:
        """Generate trading signals for all symbols.

        Args:
            data: Dict mapping symbol to DataFrame (price + features)
                 Each DataFrame has columns: open, high, low, close, volume, etc.

        Returns:
            Dict mapping symbol to {"signal": int, "position": float}
            - signal: 1 (BUY), 0 (HOLD), -1 (SELL)
            - position: ratio of available capital/position (0.0 to 1.0)
        """
        # Get all symbols
        symbols = list(data.keys())
        num_symbols = len(symbols)

        if num_symbols == 0:
            return {}

        # Calculate equal weight for each symbol
        equal_weight = 1.0 / num_symbols

        # On first call, buy all symbols equally
        if not self.has_bought:
            self.has_bought = True
            return {
                symbol: {"signal": 1, "position": equal_weight}
                for symbol in symbols
            }

        # After first buy, hold all positions
        return {
            symbol: {"signal": 0, "position": 0.0}
            for symbol in symbols
        }
