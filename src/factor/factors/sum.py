import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_sum
from src.factor.types import Factor

EPS = 1e-12

class SUM(Factor):
    """SUM factors from Alpha158."""
    
    type: str = Field(default="sum", description="The type of the factor")
    expression: str = Field(default="sump_w = ts_sum(pos_ret, w) / ts_sum(abs_ret, w), sumn_w = 1 - sump_w, sumd_w = 2 * sump_w - 1", description="The expression of the factor")
    description: str = Field(default="Sum of positive/absolute returns indicators from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=[
        "sump_5", "sumn_5", "sumd_5",
        "sump_10", "sumn_10", "sumd_10",
        "sump_20", "sumn_20", "sumd_20",
        "sump_30", "sumn_30", "sumd_30",
        "sump_60", "sumn_60", "sumd_60"
    ], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        names = []
        for window in windows:
            names.extend([f"sump_{window}", f"sumn_{window}", f"sumd_{window}"])
        self.names = names

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the SUM factors.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the SUM indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = []
        for window in _windows:
            _names.extend([f"sump_{window}", f"sumn_{window}", f"sumd_{window}"])

        ret1 = df["close"].pct_change(1)
        abs_ret1 = ret1.abs()
        pos_ret1 = np.where(ret1 < 0, 0, ret1)
        pos_ret1 = pd.Series(pos_ret1, index=df.index)
        
        for window in _windows:
            sump = divide(ts_sum(pos_ret1, window), ts_sum(abs_ret1, window) + EPS)
            
            df[f"sump_{window}"] = sump
            df[f"sumn_{window}"] = 1.0 - sump
            df[f"sumd_{window}"] = 2.0 * sump - 1.0
            
        return df[_names]
