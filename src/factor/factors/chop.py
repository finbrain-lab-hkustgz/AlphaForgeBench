import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.factor.types import Factor

EPS = 1e-12


class ChoppinessIndex(Factor):
    """Choppiness Index (CHOP).

    A non-directional regime indicator: higher values mean more sideways/choppy,
    lower values mean trending. Often used to decide when mean-reversion vs
    trend strategies should be active.

    Common formula:
      TR = max(high-low, |high-prev_close|, |low-prev_close|)
      CHOP = 100 * log10( sum(TR,n) / (max(high,n)-min(low,n)) ) / log10(n)

    Range: [0, 100]
    Outputs: chop_{period}
    """

    type: str = Field(default="chop", description="The type of the factor")
    expression: str = Field(
        default="chop = 100*log10(sum(TR,n)/(max(high,n)-min(low,n)))/log10(n)",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Choppiness Index — regime filter (range vs trend)",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["chop_14"],
        description="The returned names of the factors",
    )

    def __init__(self, periods: List[int] = [14], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"chop_{p}" for p in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        _periods = periods if periods is not None else self.periods
        _names = [f"chop_{p}" for p in _periods]

        close = df["close"].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        for p in _periods:
            sum_tr = tr.rolling(p).sum()
            hi = high.rolling(p).max()
            lo = low.rolling(p).min()
            rng = (hi - lo).replace(0, np.nan)

            ratio = (sum_tr / (rng + EPS)).replace([np.inf, -np.inf], np.nan)
            chop = 100.0 * np.log10(ratio) / np.log10(float(p))
            df[f"chop_{p}"] = chop.clip(lower=0.0, upper=100.0)

        return df[_names]

