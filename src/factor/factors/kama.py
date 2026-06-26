import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import delta, abs, ts_sum, divide, add, multiply
from src.factor.types import Factor


class KAMA(Factor):
    """Kaufman Adaptive Moving Average (KAMA).

    KAMA uses the Efficiency Ratio (ER) to dynamically adjust its smoothing
    constant between a fast EMA and a slow EMA:
      ER   = abs(delta(close, n)) / ts_sum(abs(delta(close, 1)), n)
      SC   = (ER * (fast_sc - slow_sc) + slow_sc) ^ 2
      KAMA = KAMA[-1] + SC * (close - KAMA[-1])

    Outputs: kama_{period}
    """

    type: str = Field(default="kama", description="The type of the factor")
    expression: str = Field(
        default="kama = kama[-1] + sc * (close - kama[-1]), sc = (er*(fast_sc-slow_sc)+slow_sc)^2",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Kaufman Adaptive Moving Average — self-adjusting trend filter",
        description="The description of the factor",
    )
    names: List[str] = Field(default=["kama_100"], description="The returned names of the factors")

    def __init__(self, periods: List[int] = [100], fast: int = 2, slow: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.fast = fast
        self.slow = slow
        self.names = [f"kama_{p}" for p in periods]

    async def __call__(
        self, df: pd.DataFrame,
        periods: Optional[List[int]] = None,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
    ) -> pd.DataFrame:
        _periods = periods if periods is not None else self.periods
        _fast = fast if fast is not None else self.fast
        _slow = slow if slow is not None else self.slow
        _names = [f"kama_{p}" for p in _periods]

        close = df["close"].astype(float)

        fast_sc = 2.0 / (_fast + 1)
        slow_sc = 2.0 / (_slow + 1)

        for p in _periods:
            n = len(close)
            col = f"kama_{p}"

            if n < p:
                df[col] = np.nan
                continue

            directional = abs(delta(close, p))
            noise = ts_sum(abs(delta(close, 1)), p).replace(0, np.nan)
            er = divide(directional, noise).clip(lower=0.0, upper=1.0).fillna(0.0)

            sc_raw = add(multiply(er, fast_sc - slow_sc), slow_sc)
            sc = multiply(sc_raw, sc_raw)

            sc_vals = sc.values
            close_vals = close.values
            kama = pd.Series(index=df.index, dtype=float)
            kama.iloc[p - 1] = close_vals[p - 1]

            for i in range(p, n):
                kama.iloc[i] = kama.iloc[i - 1] + sc_vals[i] * (close_vals[i] - kama.iloc[i - 1])

            df[col] = kama

        return df[_names]
