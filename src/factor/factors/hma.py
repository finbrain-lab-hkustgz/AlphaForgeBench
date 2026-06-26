import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import add, multiply
from src.factor.types import Factor


class HMA(Factor):
    """Hull Moving Average (HMA).

    HMA dramatically reduces lag compared to EMA while maintaining smoothness.

    Formula:
      WMA_half = WMA(close, period/2)
      WMA_full = WMA(close, period)
      raw      = 2 * WMA_half - WMA_full
      HMA      = WMA(raw, sqrt(period))

    Outputs: hma_{period}
    """

    type: str = Field(default="hma", description="The type of the factor")
    expression: str = Field(
        default="hma = wma(2*wma(close, period/2) - wma(close, period), sqrt(period))",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Hull Moving Average — low-lag trend-following moving average",
        description="The description of the factor",
    )
    names: List[str] = Field(default=["hma_10", "hma_30"], description="The returned names of the factors")

    def __init__(self, periods: List[int] = [10, 30], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"hma_{p}" for p in periods]

    @staticmethod
    def _wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average using linearly increasing weights [1, 2, ..., n]."""
        weights = np.arange(1, period + 1, dtype=float)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True,
        )

    async def __call__(
        self, df: pd.DataFrame,
        periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        _periods = periods if periods is not None else self.periods
        _names = [f"hma_{p}" for p in _periods]

        close = df["close"].astype(float)

        for p in _periods:
            half_p = max(1, p // 2)
            sqrt_p = max(1, int(round(p ** 0.5)))

            wma_half = self._wma(close, half_p)
            wma_full = self._wma(close, p)

            raw = add(multiply(wma_half, 2.0), multiply(wma_full, -1.0))

            df[f"hma_{p}"] = self._wma(raw, sqrt_p)

        return df[_names]
