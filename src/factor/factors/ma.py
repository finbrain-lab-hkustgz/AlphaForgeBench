import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_mean
from src.factor.types import Factor

EPS = 1e-12

class MA(Factor):
    """Moving Average (MA) factor from Alpha158 (normalized SMA)."""
    
    type: str = Field(default="ma", description="The type of the factor")
    expression: str = Field(default="ma_w = ts_mean(close, w) / close", description="The expression of the factor")
    description: str = Field(default="Normalized Moving Average (MA) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["ma_5", "ma_10", "ma_20", "ma_30", "ma_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"ma_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Moving Average (MA) factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the MA indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"ma_{window}" for window in _windows]

        for window in _windows:
            df[f"ma_{window}"] = divide(ts_mean(df["close"], window), df["close"] + EPS)
            
        return df[_names]
