import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean, ts_stddev, divide, add, multiply
from src.factor.types import Factor

class CCI(Factor):
    """Commodity Channel Index (CCI) factor."""
    
    type: str = Field(default="cci", description="The type of the factor")
    expression: str = Field(default="cci = divide(typical_price - ts_mean(typical_price, period), multiply(ts_stddev(typical_price, period), 0.015)), typical_price = divide(add(add(high, low), close), 3)", description="The expression of the factor")
    description: str = Field(default="Commodity Channel Index (CCI) indicator", description="The description of the factor")
    names: List[str] = Field(default=["cci_14"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [14], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"cci_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Commodity Channel Index (CCI) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the CCI indicator
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"cci_{period}" for period in _periods]
        
        for period in _periods:
            # Calculate typical price
            typical_price = divide(add(add(df["high"], df["low"]), df["close"]), 3)
            
            # Calculate mean and standard deviation of typical price
            mean_tp = ts_mean(typical_price, period)
            std_tp = ts_stddev(typical_price, period)
            
            # Calculate CCI
            cci = divide(typical_price - mean_tp, multiply(std_tp, 0.015))
            
            df[f"cci_{period}"] = cci
        
        return df[_names]
