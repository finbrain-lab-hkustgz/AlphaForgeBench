import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_delay, min as min_op, max as max_op
from src.factor.types import Factor

EPS = 1e-12

class RSV(Factor):
    """RSV factor from Alpha158."""
    
    type: str = Field(default="rsv", description="The type of the factor")
    expression: str = Field(default="rsv_w = (close - min(low, close.shift(w))) / (max(high, close.shift(w)) - min(low, close.shift(w)))", description="The expression of the factor")
    description: str = Field(default="Raw Stochastic Value (RSV) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["rsv_5", "rsv_10", "rsv_20", "rsv_30", "rsv_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"rsv_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the RSV factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the RSV indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"rsv_{window}" for window in _windows]

        for window in _windows:
            shift = ts_delay(df["close"], window)
            low_min = min_op(df["low"], shift)
            high_max = max_op(df["high"], shift)
            
            df[f"rsv_{window}"] = divide(df["close"] - low_min, high_max - low_min + EPS)
            
        return df[_names]
