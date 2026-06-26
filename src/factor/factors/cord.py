import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_corr, divide, ts_delay, log
from src.factor.types import Factor

EPS = 1e-12

class CORD(Factor):
    """CORD factor from Alpha158."""
    
    type: str = Field(default="cord", description="The type of the factor")
    expression: str = Field(default="cord_w = ts_corr(close/close.shift(1), log(volume/volume.shift(1)+1), w)", description="The expression of the factor")
    description: str = Field(default="Rolling Correlation between Close Change and Volume Change from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["cord_5", "cord_10", "cord_20", "cord_30", "cord_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"cord_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the CORD factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the CORD indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"cord_{window}" for window in _windows]

        close_chg = divide(df["close"], ts_delay(df["close"], 1) + EPS)
        vol_chg = log(divide(df["volume"], ts_delay(df["volume"], 1) + EPS) + 1)
        
        for window in _windows:
            df[f"cord_{window}"] = ts_corr(close_chg, vol_chg, window)
            
        return df[_names]
