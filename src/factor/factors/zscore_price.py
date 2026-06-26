import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_zscore
from src.factor.types import Factor


class ZScorePrice(Factor):
    """Rolling Z-Score of close price.

    zscore = (close - ts_mean(close, period)) / ts_stddev(close, period)

    Values > +2 indicate price far above mean (potential short signal for mean reversion).
    Values < -2 indicate price far below mean (potential long signal for mean reversion).

    Outputs: zscore_{period}
    """

    type: str = Field(default="zscore_price", description="The type of the factor")
    expression: str = Field(
        default="zscore = ts_zscore(close, period)",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Rolling Z-Score of close price for mean reversion detection",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["zscore_20"],
        description="The returned names of the factors",
    )

    def __init__(self, periods: List[int] = [20], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"zscore_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Calculate the rolling Z-Score of close price.

        Args:
            df: DataFrame with 'close' column.
            periods: Dynamic periods override. If None, uses self.periods.

        Returns:
            DataFrame with zscore columns.
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"zscore_{period}" for period in _periods]

        for period in _periods:
            df[f"zscore_{period}"] = ts_zscore(df["close"], period)

        return df[_names]
