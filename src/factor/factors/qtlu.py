import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide
from src.factor.types import Factor

EPS = 1e-12

class QTLU(Factor):
    """QTLU (Upper Quantile) factor from Alpha158."""
    
    type: str = Field(default="qtlu", description="The type of the factor")
    expression: str = Field(default="qtlu_w = (close - close.rolling(w).quantile(0.8)) / close", description="The expression of the factor")
    description: str = Field(default="Upper Quantile (80%) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["qtlu_5", "qtlu_10", "qtlu_20", "qtlu_30", "qtlu_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"qtlu_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the QTLU factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the QTLU indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"qtlu_{window}" for window in _windows]

        for window in _windows:
            quantile_val = df["close"].rolling(window=window).quantile(0.8)
            df[f"qtlu_{window}"] = divide(df["close"] - quantile_val, df["close"] + EPS)
            
        return df[_names]
