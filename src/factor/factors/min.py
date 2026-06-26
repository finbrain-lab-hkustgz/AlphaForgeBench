import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_min
from src.factor.types import Factor

EPS = 1e-12

class MIN(Factor):
    """MIN factor from Alpha158 (normalized)."""
    
    type: str = Field(default="min", description="The type of the factor")
    expression: str = Field(default="min_w = ts_min(close, w) / close", description="The expression of the factor")
    description: str = Field(default="Normalized Minimum (MIN) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["min_5", "min_10", "min_20", "min_30", "min_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"min_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the MIN factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the MIN indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"min_{window}" for window in _windows]

        for window in _windows:
            df[f"min_{window}"] = divide(ts_min(df["close"], window), df["close"] + EPS)
            
        return df[_names]
