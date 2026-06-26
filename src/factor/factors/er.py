import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import delta, abs, ts_sum, divide
from src.factor.types import Factor


class EfficiencyRatio(Factor):
    """Kaufman Efficiency Ratio (ER).

    ER measures how "directional" a move is over a window:
      ER = |price(t) - price(t-n)| / sum_{i=1..n} |price(t-i+1) - price(t-i)|

    Range: [0, 1]
      - near 1.0: clean trend (low noise / high directional efficiency)
      - near 0.0: choppy / mean-reverting noise (many reversals)

    Outputs: er_{period}
    """

    type: str = Field(default="er", description="The type of the factor")
    expression: str = Field(
        default="er = abs(delta(close, n)) / ts_sum(abs(delta(close, 1)), n)",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Kaufman Efficiency Ratio — trendiness / choppiness regime indicator",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["er_60"],
        description="The returned names of the factors",
    )

    def __init__(self, periods: List[int] = [60], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"er_{p}" for p in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Calculate Efficiency Ratio (ER).

        Args:
            df: DataFrame with 'close' column.
            periods: Dynamic periods override. If None, uses self.periods.

        Returns:
            DataFrame with er columns.
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"er_{p}" for p in _periods]

        close = df["close"].astype(float)

        for p in _periods:
            directional = abs(delta(close, p))
            noise = ts_sum(abs(delta(close, 1)), p).replace(0, np.nan)
            er = divide(directional, noise).clip(lower=0.0, upper=1.0)
            df[f"er_{p}"] = er

        return df[_names]

