import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide
from src.factor.types import Factor

EPS = 1e-12

class IMIN(Factor):
    """IMIN factor from Alpha158."""
    
    type: str = Field(default="imin", description="The type of the factor")
    expression: str = Field(default="imin_w = argmin(low, w) / w", description="The expression of the factor")
    description: str = Field(default="Rolling Argmin of Low indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["imin_5", "imin_10", "imin_20", "imin_30", "imin_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"imin_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the IMIN factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the IMIN indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"imin_{window}" for window in _windows]

        for window in _windows:
            df[f"imin_{window}"] = divide(df["low"].rolling(window=window).apply(np.argmin, raw=True), float(window))
            
        return df[_names]
