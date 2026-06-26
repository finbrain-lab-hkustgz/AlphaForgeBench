import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide, ts_sum, ts_delay
from src.factor.types import Factor

EPS = 1e-12

class VSUM(Factor):
    """VSUM (Volume Sum) factors from Alpha158."""
    
    type: str = Field(default="vsum", description="The type of the factor")
    expression: str = Field(default="vsump_w = ts_sum(pos_vol_chg, w) / ts_sum(abs_vol_chg, w), vsumn_w = 1 - vsump_w, vsumd_w = 2 * vsump_w - 1", description="The expression of the factor")
    description: str = Field(default="Volume Sum factors from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=[
        "vsump_5", "vsumn_5", "vsumd_5",
        "vsump_10", "vsumn_10", "vsumd_10",
        "vsump_20", "vsumn_20", "vsumd_20",
        "vsump_30", "vsumn_30", "vsumd_30",
        "vsump_60", "vsumn_60", "vsumd_60"
    ], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        names = []
        for window in windows:
            names.extend([f"vsump_{window}", f"vsumn_{window}", f"vsumd_{window}"])
        self.names = names

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the VSUM factors.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the VSUM indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = []
        for window in _windows:
            _names.extend([f"vsump_{window}", f"vsumn_{window}", f"vsumd_{window}"])

        vchg1 = df["volume"] - ts_delay(df["volume"], 1)
        abs_vchg1 = vchg1.abs()
        pos_vchg1 = np.where(vchg1 < 0, 0, vchg1)
        pos_vchg1 = pd.Series(pos_vchg1, index=df.index)
        
        for window in _windows:
            vsump = divide(ts_sum(pos_vchg1, window), ts_sum(abs_vchg1, window) + EPS)
            
            df[f"vsump_{window}"] = vsump
            df[f"vsumn_{window}"] = 1.0 - vsump
            df[f"vsumd_{window}"] = 2.0 * vsump - 1.0
            
        return df[_names]
