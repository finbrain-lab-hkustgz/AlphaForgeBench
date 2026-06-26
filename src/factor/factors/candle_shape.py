import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import max, min, abs as abs_op, divide, ts_mean
from src.factor.types import Factor

class CandleShape(Factor):
    """Candle shape and wick ratios.
    
    Returns:
      - upper_wick_ratio_{period}: Upper wick length / Total candle range (rolling mean)
      - lower_wick_ratio_{period}: Lower wick length / Total candle range (rolling mean)
      - body_ratio_{period}: Body length / Total candle range (rolling mean)
    """
    type: str = Field(default="candle_shape", description="The type of the factor")
    expression: str = Field(default="upper_wick_ratio = ts_mean((high - max(open, close)) / (high - low), period)", description="The expression of the factor")
    description: str = Field(default="Analyzes candlestick wicks to detect rejection.", description="The description of the factor")
    names: List[str] = Field(default=["upper_wick_ratio_1", "lower_wick_ratio_1", "body_ratio_1"], description="The returned names of the factors")

    def __init__(self, periods: List[int] = [1], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = []
        for period in periods:
            self.names.extend([f"upper_wick_ratio_{period}", f"lower_wick_ratio_{period}", f"body_ratio_{period}"])

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        _periods = periods if periods is not None else self.periods
        _names = []
        for period in _periods:
            _names.extend([f"upper_wick_ratio_{period}", f"lower_wick_ratio_{period}", f"body_ratio_{period}"])

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_p = df["open"].astype(float)
        close_p = df["close"].astype(float)
        
        candle_range = (high - low).replace(0, np.nan)
        upper_wick = high - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low
        body = abs_op(close_p - open_p)
        
        raw_upper = divide(upper_wick, candle_range).fillna(0)
        raw_lower = divide(lower_wick, candle_range).fillna(0)
        raw_body = divide(body, candle_range).fillna(0)
        
        for period in _periods:
            if period == 1:
                df[f"upper_wick_ratio_{period}"] = raw_upper
                df[f"lower_wick_ratio_{period}"] = raw_lower
                df[f"body_ratio_{period}"] = raw_body
            else:
                df[f"upper_wick_ratio_{period}"] = ts_mean(raw_upper, period)
                df[f"lower_wick_ratio_{period}"] = ts_mean(raw_lower, period)
                df[f"body_ratio_{period}"] = ts_mean(raw_body, period)
        
        return df[_names]
