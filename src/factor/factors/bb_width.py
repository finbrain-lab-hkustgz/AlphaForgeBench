import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean, ts_stddev, divide, multiply, add
from src.factor.types import Factor

EPS = 1e-12


class BBWidth(Factor):
    """Bollinger Band Width factor.

    BB Width = (Upper Band - Lower Band) / Middle Band
    Measures volatility contraction/expansion.  Low values indicate a squeeze
    (potential breakout), high values indicate expansion.

    Outputs: bb_width_{period}
    """

    type: str = Field(default="bb_width", description="The type of the factor")
    expression: str = Field(
        default="bb_width = (bb_upper - bb_lower) / bb_middle",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Bollinger Band Width — volatility squeeze/expansion indicator",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["bb_width_20"],
        description="The returned names of the factors",
    )

    def __init__(self, periods: List[int] = [20], nbdev: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.nbdev = nbdev
        self.names = [f"bb_width_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Calculate the Bollinger Band Width.

        Args:
            df: DataFrame with 'close' column.
            periods: Dynamic periods override. If None, uses self.periods.

        Returns:
            DataFrame with bb_width columns.
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"bb_width_{period}" for period in _periods]

        for period in _periods:
            middle = ts_mean(df["close"], period)
            std = ts_stddev(df["close"], period)
            upper = add(middle, multiply(std, self.nbdev))
            lower = add(middle, multiply(std, -self.nbdev))

            safe_middle = middle.replace(0, np.nan)
            df[f"bb_width_{period}"] = divide(upper - lower, safe_middle)

        return df[_names]
