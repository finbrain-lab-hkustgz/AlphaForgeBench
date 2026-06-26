import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import abs, delay, max as op_max, ts_mean
from src.factor.types import Factor


class ATRPercentile(Factor):
    """ATR percentile (rolling) for volatility regime detection.

    This project’s Factor convention is `__call__(df, periods=...)`.
    Here `periods` refers to the *percentile window(s)* in bars.

    - ATR period is fixed by `self.atr_period` (default 14). If `atr_14` does not exist,
      this factor will compute it inline.

    Outputs:
      - atr_pct_{atr_period}_{window}: percentile of current ATR within trailing window, in [0, 1]
    """

    type: str = Field(default="atr_percentile", description="The type of the factor")
    expression: str = Field(default="atr_pct = percentile_rank(atr(atr_period), window)", description="The expression of the factor")
    description: str = Field(default="ATR percentile — volatility regime (lower is calmer)", description="The description of the factor")
    names: List[str] = Field(default=["atr_pct_14_200"], description="The returned names of the factors")

    def __init__(self, atr_period: int = 14, periods: List[int] = [200], **kwargs):
        super().__init__(**kwargs)
        self.atr_period = int(atr_period)
        self.periods = [int(x) for x in periods]
        self.names = [f"atr_pct_{self.atr_period}_{w}" for w in self.periods]

    @staticmethod
    def _pct_rank_last(arr: np.ndarray) -> float:
        """Fast-ish percentile rank of the last element within arr (ignoring NaN)."""
        if arr.size == 0:
            return np.nan
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return float(np.sum(valid <= last) / valid.size)

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        _windows = [int(x) for x in (periods if periods is not None else self.periods)]
        _names = [f"atr_pct_{self.atr_period}_{w}" for w in _windows]

        atr_col = f"atr_{self.atr_period}"
        if atr_col not in df.columns:
            if {"high", "low", "close"}.issubset(df.columns):
                df[atr_col] = ts_mean(
                    op_max(
                        df["high"] - df["low"],
                        abs(df["high"] - delay(df["close"], 1)),
                        abs(df["low"] - delay(df["close"], 1)),
                    ),
                    self.atr_period,
                )
            else:
                df[atr_col] = np.nan

        atr_s = df[atr_col].astype(float)
        for w in _windows:
            col = f"atr_pct_{self.atr_period}_{w}"
            if w <= 1:
                df[col] = np.nan
            else:
                df[col] = atr_s.rolling(int(w), min_periods=max(5, int(w * 0.5))).apply(self._pct_rank_last, raw=True)

        return df[_names]

