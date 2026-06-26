"""
Factor Momentum Strategy

A portfolio strategy that allocates weights proportional to momentum factor values.
Assets with higher momentum get higher weights.

Example:
- Calculate momentum for all assets
- Normalize momentum values to positive weights
- Allocate weights proportional to momentum strength
"""

from typing import Dict, Optional
from pandas import DataFrame
from pydantic import Field
import pandas as pd
import numpy as np

from src.strategy.types import Strategy


class FactorMomentum(Strategy):
    """Factor momentum portfolio strategy.

    This strategy:
    1. Calculates momentum factor for all assets
    2. Filters out assets with negative momentum (optional)
    3. Allocates weights proportional to positive momentum values
    4. Applies weight constraints (min/max)

    Strategy Output Format:
        {
            "BTCUSDT": 0.6,   # 60% weight (higher momentum)
            "ETHUSDT": 0.4,   # 40% weight (lower momentum)
        }
    """

    name: str = Field(default="factor_momentum", description="Strategy name")
    description: str = Field(
        default="Factor momentum portfolio - weights proportional to momentum strength",
        description="Strategy description"
    )
    factor_names: list = Field(default=[], description="List of factor names used")

    # Strategy parameters
    lookback_period: int = Field(default=20, description="Lookback period for momentum calculation")
    min_weight: float = Field(default=0.0, description="Minimum weight per asset")
    max_weight: float = Field(default=0.5, description="Maximum weight per asset")
    only_positive: bool = Field(default=True, description="Only invest in assets with positive momentum")

    async def __call__(
        self,
        data: Dict[str, DataFrame],
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Generate target portfolio weights based on momentum factor.

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
        momentum_values = {}

        for symbol, df in data.items():
            try:
                # Check if we have enough data
                if len(df) < self.lookback_period:
                    # Not enough data, assign zero momentum
                    momentum_values[symbol] = 0.0
                    continue

                # Calculate momentum: (current_price - past_price) / past_price
                close_prices = df['close']
                current_price = close_prices.iloc[-1]
                past_price = close_prices.iloc[-self.lookback_period]

                momentum = (current_price - past_price) / past_price if past_price > 0 else 0.0
                momentum_values[symbol] = momentum

            except Exception as e:
                # On error, assign zero momentum
                momentum_values[symbol] = 0.0

        # Filter by momentum sign if only_positive is True
        if self.only_positive:
            filtered_momentum = {
                symbol: max(0.0, momentum)
                for symbol, momentum in momentum_values.items()
            }
        else:
            # Shift all momentum values to be positive
            min_momentum = min(momentum_values.values())
            if min_momentum < 0:
                filtered_momentum = {
                    symbol: momentum - min_momentum
                    for symbol, momentum in momentum_values.items()
                }
            else:
                filtered_momentum = momentum_values

        # Calculate total momentum (for normalization)
        total_momentum = sum(filtered_momentum.values())

        # Allocate weights proportional to momentum
        if total_momentum > 0:
            target_weights = {
                symbol: momentum / total_momentum
                for symbol, momentum in filtered_momentum.items()
            }
        else:
            # No positive momentum, equal weight or stay in cash
            target_weights = {symbol: 0.0 for symbol in data.keys()}

        # Apply weight constraints
        for symbol in target_weights.keys():
            weight = target_weights[symbol]
            # Clip to [min_weight, max_weight]
            target_weights[symbol] = max(self.min_weight, min(weight, self.max_weight))

        # Renormalize if needed (after applying constraints)
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            target_weights = {
                symbol: weight / total_weight
                for symbol, weight in target_weights.items()
            }

        return target_weights
