import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_max
from src.factor.types import Factor

EPS = 1e-12

class MAX(Factor):
    """MAX factor from Alpha158 (normalized)."""
    
    type: str = Field(default="max", description="The type of the factor")
    expression: str = Field(default="max_w = ts_max(close, w) / close", description="The expression of the factor")
    description: str = Field(default="Normalized Maximum (MAX) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["max_5", "max_10", "max_20", "max_30", "max_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"max_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the MAX factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the MAX indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"max_{window}" for window in _windows]

        for window in _windows:
            df[f"max_{window}"] = divide(ts_max(df["close"], window), df["close"] + EPS)
            
        return df[_names]
