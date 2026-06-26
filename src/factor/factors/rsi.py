import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean, delta, abs, divide, add
from src.factor.types import Factor

class RSI(Factor):
    """Relative Strength Index (RSI) factor."""
    
    type: str = Field(default="rsi", description="The type of the factor")
    expression: str = Field(default="rsi = 100 - 100 / (1 + rs), rs = ts_mean(gain, period) / ts_mean(loss, period)", description="The expression of the factor")
    description: str = Field(default="Relative Strength Index (RSI) indicator", description="The description of the factor")
    names: List[str] = Field(default=["rsi_14"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [14], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"rsi_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the RSI indicator
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"rsi_{period}" for period in _periods]
        
        for period in _periods:
            # Calculate price change
            delta_price = delta(df["close"], 1)
            
            # Separate gains and losses
            gain = delta_price.copy()
            loss = delta_price.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and loss
            avg_gain = ts_mean(gain, period)
            avg_loss = ts_mean(loss, period)
            
            # Calculate RS and RSI
            rs = divide(avg_gain, avg_loss)
            rsi = 100 - divide(100, add(rs, 1))
            
            df[f"rsi_{period}"] = rsi
        
        return df[_names]
