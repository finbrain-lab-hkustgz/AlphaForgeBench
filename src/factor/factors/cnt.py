import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide
from src.factor.types import Factor

EPS = 1e-12

class CNT(Factor):
    """CNT (Count) factors from Alpha158."""
    
    type: str = Field(default="cnt", description="The type of the factor")
    expression: str = Field(default="cntp_w = count(ret > 0, w)/w, cntn_w = count(ret < 0, w)/w, cntd_w = cntp_w - cntn_w", description="The expression of the factor")
    description: str = Field(default="Count of positive/negative returns indicators from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=[
        "cntp_5", "cntn_5", "cntd_5",
        "cntp_10", "cntn_10", "cntd_10",
        "cntp_20", "cntn_20", "cntd_20",
        "cntp_30", "cntn_30", "cntd_30",
        "cntp_60", "cntn_60", "cntd_60"
    ], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        names = []
        for window in windows:
            names.extend([f"cntp_{window}", f"cntn_{window}", f"cntd_{window}"])
        self.names = names

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the CNT factors.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the CNT indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = []
        for window in _windows:
            _names.extend([f"cntp_{window}", f"cntn_{window}", f"cntd_{window}"])

        ret1 = df["close"].pct_change(1)
        
        for window in _windows:
            cntp = ret1.gt(0).rolling(window=window).sum() / float(window)
            cntn = ret1.lt(0).rolling(window=window).sum() / float(window)
            
            df[f"cntp_{window}"] = cntp
            df[f"cntn_{window}"] = cntn
            df[f"cntd_{window}"] = cntp - cntn
            
        return df[_names]
