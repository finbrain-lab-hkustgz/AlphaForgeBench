import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import delta, abs, reverse, ts_mean, divide, multiply, max
from src.factor.types import Factor

EPS = 1e-12


class ADX(Factor):
    """Average Directional Index (ADX) factor.

    Measures trend strength regardless of direction.
    ADX > 25 indicates strong trend, ADX < 20 indicates ranging market.

    Outputs: adx_{period}, plus_di_{period}, minus_di_{period}
    """

    type: str = Field(default="adx", description="The type of the factor")
    expression: str = Field(
        default="adx = ts_mean(dx, period), dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Average Directional Index (ADX) — trend strength indicator",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["adx_14", "plus_di_14", "minus_di_14"],
        description="The returned names of the factors",
    )

    def __init__(self, periods: List[int] = [14], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = []
        for period in periods:
            self.names.extend([f"adx_{period}", f"plus_di_{period}", f"minus_di_{period}"])

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Calculate the Average Directional Index (ADX).

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.
            periods: Dynamic periods override. If None, uses self.periods.

        Returns:
            DataFrame with adx, plus_di, minus_di columns.
        """
        _periods = periods if periods is not None else self.periods
        _names = []
        for period in _periods:
            _names.extend([f"adx_{period}", f"plus_di_{period}", f"minus_di_{period}"])

        for period in _periods:
            # +DM / -DM (simplified: no cross-comparison, consistent with common implementations)
            plus_dm = delta(df["high"], 1)
            minus_dm = reverse(delta(df["low"], 1))

            # Clip negative values to 0
            plus_dm = plus_dm.clip(lower=0)
            minus_dm = minus_dm.clip(lower=0)

            # True Range
            tr = max(
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            )

            # Smoothed averages
            atr = ts_mean(tr, period)
            atr_safe = atr.replace(0, np.nan)

            # Directional Indicators
            plus_di = multiply(divide(ts_mean(plus_dm, period), atr_safe), 100)
            minus_di = multiply(divide(ts_mean(minus_dm, period), atr_safe), 100)

            # DX and ADX
            di_sum = (plus_di + minus_di).replace(0, np.nan)
            dx = multiply(divide(abs(plus_di - minus_di), di_sum), 100)
            adx = ts_mean(dx, period)

            df[f"adx_{period}"] = adx
            df[f"plus_di_{period}"] = plus_di
            df[f"minus_di_{period}"] = minus_di

        return df[_names]
