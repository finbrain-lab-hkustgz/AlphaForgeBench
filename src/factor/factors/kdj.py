import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_min, ts_max, divide, multiply, add, ts_mean
from src.factor.types import Factor

class KDJ(Factor):
    """KDJ (Stochastic Oscillator) factor."""
    
    type: str = Field(default="kdj", description="The type of the factor")
    expression: str = Field(default="stoch_k = ts_mean(rsv, slowk_period), stoch_d = ts_mean(stoch_k, slowd_period), rsv = multiply(divide(close - ts_min(low, period), ts_max(high, period) - ts_min(low, period)), 100)", description="The expression of the factor")
    description: str = Field(default="KDJ (Stochastic Oscillator) indicator", description="The description of the factor")
    names: List[str] = Field(default=["stoch_k_14", "stoch_d_14"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [14], slowk_period: int = 3, slowd_period: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period
        self.names = []
        for period in periods:
            self.names.extend([f"stoch_k_{period}", f"stoch_d_{period}"])

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the KDJ (Stochastic Oscillator) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the KDJ indicator (stoch_k, stoch_d)
        """
        _periods = periods if periods is not None else self.periods
        _names = []
        for period in _periods:
            _names.extend([f"stoch_k_{period}", f"stoch_d_{period}"])
        
        for period in _periods:
            # Calculate RSV (Raw Stochastic Value)
            lowest_low = ts_min(df["low"], period)
            highest_high = ts_max(df["high"], period)
            numerator = df["close"] - lowest_low
            denominator = highest_high - lowest_low
            rsv = multiply(divide(numerator, denominator), 100)
            
            # Calculate K line (fast stochastic)
            k = ts_mean(rsv, self.slowk_period)
            
            # Calculate D line (slow stochastic, smoothed K)
            d = ts_mean(k, self.slowd_period)
            
            df[f"stoch_k_{period}"] = k
            df[f"stoch_d_{period}"] = d
        
        return df[_names]
