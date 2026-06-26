import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean, ts_stddev, add, multiply
from src.factor.types import Factor

class BB(Factor):
    """Bollinger Bands (BB) factor."""
    
    type: str = Field(default="bb", description="The type of the factor")
    expression: str = Field(default="bb_upper = add(ts_mean(close, period), multiply(ts_stddev(close, period), nbdev)), bb_middle = ts_mean(close, period), bb_lower = add(ts_mean(close, period), multiply(ts_stddev(close, period), -nbdev))", description="The expression of the factor")
    description: str = Field(default="Bollinger Bands (BB) indicator", description="The description of the factor")
    names: List[str] = Field(default=["bb_upper_20", "bb_middle_20", "bb_lower_20"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [20], nbdev: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.nbdev = nbdev
        self.names = []
        for period in periods:
            self.names.extend([f"bb_upper_{period}", f"bb_middle_{period}", f"bb_lower_{period}"])

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Bollinger Bands (BB) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the BB indicator (upper, middle, lower bands)
        """
        _periods = periods if periods is not None else self.periods
        _names = []
        for period in _periods:
            _names.extend([f"bb_upper_{period}", f"bb_middle_{period}", f"bb_lower_{period}"])
        
        for period in _periods:
            middle = ts_mean(df["close"], period)
            std = ts_stddev(df["close"], period)
            
            df[f"bb_middle_{period}"] = middle
            df[f"bb_upper_{period}"] = add(middle, multiply(std, self.nbdev))
            df[f"bb_lower_{period}"] = add(middle, multiply(std, -self.nbdev))
        
        return df[_names]
