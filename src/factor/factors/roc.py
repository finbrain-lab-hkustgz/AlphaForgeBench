import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_delay
from src.factor.types import Factor

EPS = 1e-12

class ROC(Factor):
    """Rate of Change (ROC) factor from Alpha158."""
    
    type: str = Field(default="roc", description="The type of the factor")
    expression: str = Field(default="roc_w = close.shift(w) / close", description="The expression of the factor")
    description: str = Field(default="Rate of Change (ROC) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["roc_5", "roc_10", "roc_20", "roc_30", "roc_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"roc_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Rate of Change (ROC) factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the ROC indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"roc_{window}" for window in _windows]

        for window in _windows:
            df[f"roc_{window}"] = divide(ts_delay(df["close"], window), df["close"] + EPS)
            
        return df[_names]
