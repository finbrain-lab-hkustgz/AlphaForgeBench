import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_std_dev, ts_mean, ts_delay
from src.factor.types import Factor

EPS = 1e-12

class WVMA(Factor):
    """WVMA (Weighted Volume Moving Average) factor from Alpha158."""
    
    type: str = Field(default="wvma", description="The type of the factor")
    expression: str = Field(default="wvma_w = ts_std_dev(abs(ret)*vol, w) / ts_mean(abs(ret)*vol, w)", description="The expression of the factor")
    description: str = Field(default="Weighted Volume Moving Average indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["wvma_5", "wvma_10", "wvma_20", "wvma_30", "wvma_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"wvma_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the WVMA factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the WVMA indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"wvma_{window}" for window in _windows]

        ret_vol = np.abs(divide(df["close"], ts_delay(df["close"], 1) + EPS) - 1) * df["volume"]
        
        for window in _windows:
            std = ts_std_dev(ret_vol, window)
            mean = ts_mean(ret_vol, window)
            df[f"wvma_{window}"] = divide(std, mean + EPS).fillna(0)
            
        return df[_names]
