import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_std_dev
from src.factor.types import Factor

EPS = 1e-12

class STD(Factor):
    """Standard Deviation (STD) factor from Alpha158 (normalized)."""
    
    type: str = Field(default="std", description="The type of the factor")
    expression: str = Field(default="std_w = ts_std_dev(close, w) / close", description="The expression of the factor")
    description: str = Field(default="Normalized Standard Deviation (STD) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["std_5", "std_10", "std_20", "std_30", "std_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"std_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Standard Deviation (STD) factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the STD indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"std_{window}" for window in _windows]

        for window in _windows:
            df[f"std_{window}"] = divide(ts_std_dev(df["close"], window), df["close"] + EPS)
            
        return df[_names]
