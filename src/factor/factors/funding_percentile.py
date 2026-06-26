import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.factor.types import Factor

FUNDING_COL = "funding_rate"


class FundingPercentile(Factor):
    """Funding rate percentile (rolling).

    Requires a `funding_rate` column in df. If absent, all outputs are NaN.

    Outputs:
      - funding_pct_{period}: percentile rank of the current funding rate
        within a trailing window, in [0, 1].
    """

    type: str = Field(default="funding_percentile", description="The type of the factor")
    expression: str = Field(default="funding_pct = percentile_rank(funding_rate, period)", description="The expression of the factor")
    description: str = Field(default="Funding rate percentile — sentiment/crowding regime signal", description="The description of the factor")
    names: List[str] = Field(default=["funding_pct_500"], description="The returned names of the factors")

    def __init__(self, periods: List[int] = [500], **kwargs):
        super().__init__(**kwargs)
        self.periods = [int(x) for x in periods]
        self.names = [f"funding_pct_{w}" for w in self.periods]

    @staticmethod
    def _pct_rank_last(arr: np.ndarray) -> float:
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return float(np.sum(valid <= last) / valid.size)

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        _windows = [int(x) for x in (periods if periods is not None else self.periods)]
        _names = [f"funding_pct_{w}" for w in _windows]

        if FUNDING_COL not in df.columns:
            for w in _windows:
                df[f"funding_pct_{w}"] = np.nan
            return df[_names]

        f = df[FUNDING_COL].astype(float)
        for w in _windows:
            if w <= 1:
                df[f"funding_pct_{w}"] = np.nan
            else:
                df[f"funding_pct_{w}"] = f.rolling(int(w), min_periods=max(5, int(w * 0.5))).apply(
                    self._pct_rank_last, raw=True
                )

        return df[_names]
