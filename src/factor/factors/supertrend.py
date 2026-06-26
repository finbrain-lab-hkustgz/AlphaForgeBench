import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import abs, delay, max, ts_mean, add, multiply, divide
from src.factor.types import Factor


class Supertrend(Factor):
    """Supertrend indicator.

    ATR-based trailing band that flips between support (bullish) and
    resistance (bearish). Direction changes are clean and definitive.

    Formula:
      hl2         = (high + low) / 2
      upper_band  = hl2 + multiplier * ATR(period)
      lower_band  = hl2 - multiplier * ATR(period)
      direction flips when close crosses a band (stateful ratchet).

    Outputs per (period, multiplier) pair:
      supertrend_{period}_{mult10}        — the trailing band value
      supertrend_dir_{period}_{mult10}    — direction: +1 bullish, -1 bearish
    """

    type: str = Field(default="supertrend", description="The type of the factor")
    expression: str = Field(
        default="supertrend = hl2 +/- multiplier * ts_mean(tr, period)",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Supertrend — ATR-based adaptive trend direction indicator",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["supertrend_10_30", "supertrend_dir_10_30"],
        description="The returned names of the factors",
    )

    def __init__(
        self,
        periods: List[int] = [10],
        multipliers: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.periods = periods
        self.multipliers = multipliers or [3.0]
        self.names = []
        for p in self.periods:
            for m in self.multipliers:
                tag = f"{p}_{int(m * 10)}"
                self.names.extend([f"supertrend_{tag}", f"supertrend_dir_{tag}"])

    async def __call__(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None,
        multipliers: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        _periods = periods if periods is not None else self.periods
        _multipliers = multipliers if multipliers is not None else self.multipliers

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        out_names: List[str] = []

        tr = max(high - low, abs(high - delay(close, 1)), abs(low - delay(close, 1)))

        hl2 = divide(add(high, low), 2.0)

        def _wilder_atr(tr_series: pd.Series, period: int) -> pd.Series:
            """Wilder ATR / RMA(TR, period) to match TradingView/Binance Supertrend."""
            tr_vals = tr_series.astype(float).values
            n = len(tr_vals)
            atr_vals = np.full(n, np.nan, dtype=float)
            p = int(period)
            if p <= 0 or n < p:
                return pd.Series(atr_vals, index=tr_series.index)
            atr_vals[p - 1] = float(np.nanmean(tr_vals[:p]))
            for i in range(p, n):
                prev = atr_vals[i - 1]
                atr_vals[i] = (prev * (p - 1) + tr_vals[i]) / p
            return pd.Series(atr_vals, index=tr_series.index)

        for period in _periods:
            # Binance/TradingView 的 Supertrend 通常使用 Wilder ATR（RMA），不是 SMA。
            atr = _wilder_atr(tr, int(period))

            for mult in _multipliers:
                tag = f"{period}_{int(mult * 10)}"

                upper_raw = add(hl2, multiply(atr, mult))
                lower_raw = add(hl2, multiply(atr, -mult))

                upper_vals = upper_raw.values.copy()
                lower_vals = lower_raw.values.copy()
                close_vals = close.values
                n = len(close_vals)

                st_val = np.full(n, np.nan)
                st_dir = np.full(n, np.nan)

                # 第一个可用 ATR 通常在 period-1
                # 注意：本文件从 src.operator 导入了 max（Series op），不能用 Python 内置 max()
                start = int(period) - 1
                if start < 0:
                    start = 0
                if start < n:
                    st_dir[start] = 1.0
                    st_val[start] = lower_vals[start]

                for i in range(start + 1, n):
                    if np.isnan(upper_vals[i]):
                        continue

                    if not np.isnan(lower_vals[i - 1]) and lower_vals[i] < lower_vals[i - 1] and close_vals[i - 1] > lower_vals[i - 1]:
                        lower_vals[i] = lower_vals[i - 1]
                    if not np.isnan(upper_vals[i - 1]) and upper_vals[i] > upper_vals[i - 1] and close_vals[i - 1] < upper_vals[i - 1]:
                        upper_vals[i] = upper_vals[i - 1]

                    prev_dir = st_dir[i - 1] if not np.isnan(st_dir[i - 1]) else 1.0

                    if prev_dir == 1.0:
                        if close_vals[i] < lower_vals[i]:
                            st_dir[i] = -1.0
                            st_val[i] = upper_vals[i]
                        else:
                            st_dir[i] = 1.0
                            st_val[i] = lower_vals[i]
                    else:
                        if close_vals[i] > upper_vals[i]:
                            st_dir[i] = 1.0
                            st_val[i] = lower_vals[i]
                        else:
                            st_dir[i] = -1.0
                            st_val[i] = upper_vals[i]

                col_val = f"supertrend_{tag}"
                col_dir = f"supertrend_dir_{tag}"
                df[col_val] = st_val
                df[col_dir] = st_dir
                out_names.extend([col_val, col_dir])

        return df[out_names]
