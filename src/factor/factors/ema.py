import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean, multiply, divide, add
from src.factor.types import Factor

class EMA(Factor):
    """Exponential Moving Average (EMA) factor."""
    
    type: str = Field(default="ema", description="The type of the factor")
    expression: str = Field(default="ema = ema(close, period)", description="The expression of the factor")
    description: str = Field(default="Exponential Moving Average (EMA) indicator", description="The description of the factor")
    names: List[str] = Field(default=["ema_20", "ema_50"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [20, 50], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"ema_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Exponential Moving Average (EMA) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the EMA indicator
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"ema_{period}" for period in _periods]
        
        for period in _periods:
            # EMA calculation: EMA = (close - prev_EMA) * multiplier + prev_EMA
            # multiplier = 2 / (period + 1)
            multiplier = 2.0 / (period + 1)
            ema = pd.Series(index=df.index, dtype=float)
            ema.iloc[0] = df["close"].iloc[0]
            
            for i in range(1, len(df)):
                ema.iloc[i] = (df["close"].iloc[i] - ema.iloc[i-1]) * multiplier + ema.iloc[i-1]
            
            df[f"ema_{period}"] = ema
        
        return df[_names]
