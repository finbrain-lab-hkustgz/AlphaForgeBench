import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.factor.types import Factor

PRICE_COL = "close"


class PriceAcceleration(Factor):
    """Second derivative of price: rate-of-change of rate-of-change.

    Detects momentum acceleration / deceleration for regime detection.
    Positive = price accelerating upward, negative = accelerating downward.

    Outputs:
      - price_accel_{period}: (ROC_t - ROC_{t-period}) / period, normalized by price.
    """

    type: str = Field(default="price_acceleration", description="The type of the factor")
    expression: str = Field(default="price_accel = diff(roc(close, period), period)", description="The expression of the factor")
    description: str = Field(default="Price acceleration — second derivative for regime detection", description="The description of the factor")
    names: List[str] = Field(default=["price_accel_14"], description="The returned names of the factors")

    def __init__(self, periods: List[int] = [14], **kwargs):
        super().__init__(**kwargs)
        self.periods = [int(x) for x in periods]
        self.names = [f"price_accel_{w}" for w in self.periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        _windows = [int(x) for x in (periods if periods is not None else self.periods)]
        _names = [f"price_accel_{w}" for w in _windows]

        if PRICE_COL not in df.columns:
            for w in _windows:
                df[f"price_accel_{w}"] = np.nan
            return df[_names]

        p = df[PRICE_COL].astype(float)
        for w in _windows:
            col = f"price_accel_{w}"
            if w <= 1:
                df[col] = np.nan
            else:
                roc = p.pct_change(w)
                df[col] = roc.diff(w)

        return df[_names]
