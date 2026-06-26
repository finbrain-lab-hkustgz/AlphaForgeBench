import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide
from src.factor.types import Factor

EPS = 1e-12

class IMXD(Factor):
    """IMXD factor from Alpha158."""
    
    type: str = Field(default="imxd", description="The type of the factor")
    expression: str = Field(default="imxd_w = (argmax(high, w) - argmin(low, w)) / w", description="The expression of the factor")
    description: str = Field(default="Rolling Argmax/Argmin difference indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["imxd_5", "imxd_10", "imxd_20", "imxd_30", "imxd_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"imxd_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the IMXD factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the IMXD indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"imxd_{window}" for window in _windows]

        for window in _windows:
            high_argmax = df["high"].rolling(window=window).apply(np.argmax, raw=True)
            low_argmin = df["low"].rolling(window=window).apply(np.argmin, raw=True)
            df[f"imxd_{window}"] = divide(high_argmax - low_argmin, float(window))
            
        return df[_names]
