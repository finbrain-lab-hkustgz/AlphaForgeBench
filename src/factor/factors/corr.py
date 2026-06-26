import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_corr, log
from src.factor.types import Factor

EPS = 1e-12

class CORR(Factor):
    """CORR factor from Alpha158."""
    
    type: str = Field(default="corr", description="The type of the factor")
    expression: str = Field(default="corr_w = ts_corr(close, log(volume + 1), w)", description="The expression of the factor")
    description: str = Field(default="Rolling Correlation between Close and Log Volume from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["corr_5", "corr_10", "corr_20", "corr_30", "corr_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"corr_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the CORR factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the CORR indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"corr_{window}" for window in _windows]

        log_vol = log(df["volume"] + 1)
        for window in _windows:
            df[f"corr_{window}"] = ts_corr(df["close"], log_vol, window)
            
        return df[_names]
