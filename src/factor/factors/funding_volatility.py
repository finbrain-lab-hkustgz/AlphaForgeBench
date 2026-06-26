import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.factor.types import Factor

FUNDING_COL = "funding_rate"


class FundingVolatility(Factor):
    """Rolling standard deviation of funding_rate.

    Measures funding rate instability; high values indicate unreliable carry.

    Outputs:
      - funding_vol_{period}: rolling std of the funding_rate column.
    """

    type: str = Field(default="funding_volatility", description="The type of the factor")
    expression: str = Field(default="funding_vol = std(funding_rate, period)", description="The expression of the factor")
    description: str = Field(default="Funding rate volatility — carry reliability signal", description="The description of the factor")
    names: List[str] = Field(default=["funding_vol_500"], description="The returned names of the factors")

    def __init__(self, periods: List[int] = [500], **kwargs):
        super().__init__(**kwargs)
        self.periods = [int(x) for x in periods]
        self.names = [f"funding_vol_{w}" for w in self.periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        _windows = [int(x) for x in (periods if periods is not None else self.periods)]
        _names = [f"funding_vol_{w}" for w in _windows]

        if FUNDING_COL not in df.columns:
            for w in _windows:
                df[f"funding_vol_{w}"] = np.nan
            return df[_names]

        f = df[FUNDING_COL].astype(float)
        for w in _windows:
            col = f"funding_vol_{w}"
            if w <= 1:
                df[col] = np.nan
            else:
                df[col] = f.rolling(int(w), min_periods=max(5, int(w * 0.5))).std()

        return df[_names]
