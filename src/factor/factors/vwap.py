import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import add, divide, multiply, ts_sum
from src.factor.types import Factor

EPS = 1e-12


class VWAP(Factor):
    """Rolling Volume-Weighted Average Price (VWAP) factor.

    VWAP = sum(typical_price * volume, period) / sum(volume, period)
    where typical_price = (high + low + close) / 3

    Outputs: vwap_{period}
    """

    type: str = Field(default="vwap", description="The type of the factor")
    expression: str = Field(
        default="vwap = ts_sum(typical_price * volume, period) / ts_sum(volume, period)",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Rolling Volume-Weighted Average Price (VWAP)",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["vwap_20"],
        description="The returned names of the factors",
    )

    def __init__(self, periods: List[int] = [20], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"vwap_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Calculate the rolling VWAP.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns.
            periods: Dynamic periods override. If None, uses self.periods.

        Returns:
            DataFrame with vwap columns.
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"vwap_{period}" for period in _periods]

        typical_price = divide(add(add(df["high"], df["low"]), df["close"]), 3)

        for period in _periods:
            tp_vol = multiply(typical_price, df["volume"])
            vol_sum = ts_sum(df["volume"], period).replace(0, np.nan)
            df[f"vwap_{period}"] = divide(ts_sum(tp_vol, period), vol_sum)

        return df[_names]
