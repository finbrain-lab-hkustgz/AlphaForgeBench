"""
Factor Equal Weight Strategy

A simple portfolio strategy that uses factor values to filter assets,
then allocates equal weight to selected assets.

Example:
- Calculate momentum factor for all assets
- Select top N assets by factor value
- Allocate equal weight (1/N) to selected assets
"""

from typing import Dict, Optional
from pandas import DataFrame
from pydantic import Field
import pandas as pd
import numpy as np

from src.strategy.types import Strategy


class FactorEqualWeight(Strategy):
    """Factor-based equal weight portfolio strategy.

    This strategy:
    1. Calculates a momentum factor (or uses provided factors)
    2. Ranks assets by factor value
    3. Selects top N assets
    4. Allocates equal weight to selected assets

    Strategy Output Format:
        {
            "BTCUSDT": 0.5,   # 50% weight
            "ETHUSDT": 0.5,   # 50% weight
        }
    """

    name: str = Field(default="factor_equal_weight", description="Strategy name")
    description: str = Field(
        default="Factor-driven equal weight portfolio - selects top assets by factor value",
        description="Strategy description"
    )
    factor_names: list = Field(default=[], description="List of factor names used")

    # Strategy parameters
    top_n: int = Field(default=2, description="Number of top assets to select")
    lookback_period: int = Field(default=20, description="Lookback period for momentum calculation")
    min_weight: float = Field(default=0.0, description="Minimum weight per asset")
    max_weight: float = Field(default=1.0, description="Maximum weight per asset")

    async def __call__(
        self,
        data: Dict[str, DataFrame],
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Generate target portfolio weights based on factor values.

        Args:
            data: Dict mapping symbol to DataFrame (price + features)
                 Each DataFrame has columns: open, high, low, close, volume, etc.
            current_weights: Current portfolio weights (optional, for reference)

        Returns:
            Dict mapping symbol to target weight (sum <= 1.0)
        """
        if current_weights is None:
            current_weights = {symbol: 0.0 for symbol in data.keys()}

        # Calculate momentum factor for each asset
        factor_values = {}

        for symbol, df in data.items():
            try:
                # Check if we have enough data
                if len(df) < self.lookback_period:
                    # Not enough data, assign neutral factor
                    factor_values[symbol] = 0.0
                    continue

                # Calculate momentum: (current_price - past_price) / past_price
                close_prices = df['close']
                current_price = close_prices.iloc[-1]
                past_price = close_prices.iloc[-self.lookback_period]

                momentum = (current_price - past_price) / past_price if past_price > 0 else 0.0
                factor_values[symbol] = momentum

            except Exception as e:
                # On error, assign neutral factor
                factor_values[symbol] = 0.0

        # Rank assets by factor value
        ranked_symbols = sorted(
            factor_values.keys(),
            key=lambda s: factor_values[s],
            reverse=True  # Higher momentum is better
        )

        # Select top N assets
        selected_symbols = ranked_symbols[:self.top_n]

        # Calculate equal weight
        if len(selected_symbols) > 0:
            equal_weight = 1.0 / len(selected_symbols)
            # Apply min/max weight constraints
            equal_weight = max(self.min_weight, min(equal_weight, self.max_weight))
        else:
            equal_weight = 0.0

        # Build target weights
        target_weights = {}
        for symbol in data.keys():
            if symbol in selected_symbols:
                target_weights[symbol] = equal_weight
            else:
                target_weights[symbol] = 0.0

        # Normalize to ensure sum <= 1.0
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            target_weights = {
                symbol: weight / total_weight
                for symbol, weight in target_weights.items()
            }

        return target_weights
