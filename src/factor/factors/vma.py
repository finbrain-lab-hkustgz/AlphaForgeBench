import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_mean
from src.factor.types import Factor

EPS = 1e-12

class VMA(Factor):
    """Volume Moving Average (VMA) factor from Alpha158 (normalized)."""
    
    type: str = Field(default="vma", description="The type of the factor")
    expression: str = Field(default="vma_w = ts_mean(volume, w) / volume", description="The expression of the factor")
    description: str = Field(default="Normalized Volume Moving Average (VMA) indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["vma_5", "vma_10", "vma_20", "vma_30", "vma_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"vma_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the VMA factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the VMA indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"vma_{window}" for window in _windows]

        for window in _windows:
            df[f"vma_{window}"] = divide(ts_mean(df["volume"], window), df["volume"] + EPS)
            
        return df[_names]
