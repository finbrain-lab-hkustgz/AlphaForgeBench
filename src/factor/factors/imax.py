import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import divide
from src.factor.types import Factor

EPS = 1e-12

class IMAX(Factor):
    """IMAX factor from Alpha158."""
    
    type: str = Field(default="imax", description="The type of the factor")
    expression: str = Field(default="imax_w = argmax(high, w) / w", description="The expression of the factor")
    description: str = Field(default="Rolling Argmax of High indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["imax_5", "imax_10", "imax_20", "imax_30", "imax_60"], description="The returned names of the factors")
    
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 60], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.names = [f"imax_{window}" for window in windows]

    async def __call__(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the IMAX factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            windows (Optional[List[int]]): Dynamic windows override. If None, uses self.windows.
            
        Returns:
            pd.DataFrame: The DataFrame containing the IMAX indicators
        """
        _windows = windows if windows is not None else self.windows
        _names = [f"imax_{window}" for window in _windows]

        for window in _windows:
            df[f"imax_{window}"] = divide(df["high"].rolling(window=window).apply(np.argmax, raw=True), float(window))
            
        return df[_names]
