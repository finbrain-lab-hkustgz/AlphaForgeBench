import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_rank
from src.factor.types import Factor

EPS = 1e-12

class RANK(Factor):
    """RANK factor from Alpha158."""
    
    type: str = Field(default="rank", description="The type of the factor")
    expression: str = Field(default="rank_w = ts_rank(close, w) / w", description="The expression of the factor")
    description: str = Field(default="Rolling Rank indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["rank_5", "rank_10", "rank_20", "rank_30", "rank_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"rank_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the RANK factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the RANK indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"rank_{window}" for window in _windows]

        for window in _windows:
            df[f"rank_{window}"] = divide(ts_rank(df["close"], window), float(window))
            
        return df[_names]
