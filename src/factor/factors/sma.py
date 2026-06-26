import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean
from src.factor.types import Factor

class SMA(Factor):
    """Simple Moving Average (SMA) factor."""
    
    type: str = Field(default="sma", description="The type of the factor")
    expression: str = Field(default="sma = ts_mean(close, period)", description="The expression of the factor")
    description: str = Field(default="Simple Moving Average (SMA) indicator", description="The description of the factor")
    names: List[str] = Field(default=["sma_20", "sma_50"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [20, 50], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"sma_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Simple Moving Average (SMA) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the SMA indicator
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"sma_{period}" for period in _periods]
        
        for period in _periods:
            df[f"sma_{period}"] = ts_mean(df["close"], period)
        
        return df[_names]
