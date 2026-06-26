import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide
from src.factor.types import Factor

EPS = 1e-12

class QTLD(Factor):
    """QTLD (Lower Quantile) factor from Alpha158."""
    
    type: str = Field(default="qtld", description="The type of the factor")
    expression: str = Field(default="qtld_w = (close - close.rolling(w).quantile(0.2)) / close", description="The expression of the factor")
    description: str = Field(default="Lower Quantile (20%) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["qtld_5", "qtld_10", "qtld_20", "qtld_30", "qtld_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"qtld_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the QTLD factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the QTLD indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"qtld_{window}" for window in _windows]

        for window in _windows:
            quantile_val = df["close"].rolling(window=window).quantile(0.2)
            df[f"qtld_{window}"] = divide(df["close"] - quantile_val, df["close"] + EPS)
            
        return df[_names]
