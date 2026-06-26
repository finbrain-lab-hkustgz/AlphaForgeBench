import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_delay, multiply
from src.factor.types import Factor

EPS = 1e-12

class BETA(Factor):
    """BETA factor from Alpha158."""
    
    type: str = Field(default="beta", description="The type of the factor")
    expression: str = Field(default="beta_w = (close.shift(w) - close) / (w * close)", description="The expression of the factor")
    description: str = Field(default="BETA indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["beta_5", "beta_10", "beta_20", "beta_30", "beta_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"beta_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the BETA factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the BETA indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"beta_{window}" for window in _windows]

        for window in _windows:
            numerator = ts_delay(df["close"], window) - df["close"]
            denominator = multiply(df["close"], window)
            df[f"beta_{window}"] = divide(numerator, denominator + EPS)
            
        return df[_names]
