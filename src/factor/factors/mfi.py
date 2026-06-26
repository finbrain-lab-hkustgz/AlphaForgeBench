import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean, delta, abs as abs_op, divide, multiply, add
from src.factor.types import Factor

class MFI(Factor):
    """Money Flow Index (MFI) factor."""
    
    type: str = Field(default="mfi", description="The type of the factor")
    expression: str = Field(default="mfi = 100 - 100 / (1 + money_flow_ratio), money_flow_ratio = ts_mean(positive_money_flow, period) / ts_mean(negative_money_flow, period)", description="The expression of the factor")
    description: str = Field(default="Money Flow Index (MFI) indicator", description="The description of the factor")
    names: List[str] = Field(default=["mfi_14"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [14], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"mfi_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Money Flow Index (MFI) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price and volume data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the MFI indicator
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"mfi_{period}" for period in _periods]
        
        for period in _periods:
            # Calculate typical price
            typical_price = divide(add(add(df["high"], df["low"]), df["close"]), 3)
            
            # Calculate money flow
            money_flow = multiply(typical_price, df["volume"])
            
            # Calculate price change direction
            delta_price = delta(typical_price, 1)
            
            # Separate positive and negative money flow
            positive_mf = money_flow.copy()
            negative_mf = money_flow.copy()
            positive_mf[delta_price <= 0] = 0
            negative_mf[delta_price >= 0] = 0
            negative_mf = abs_op(negative_mf)
            
            # Calculate average positive and negative money flow
            avg_positive_mf = ts_mean(positive_mf, period)
            avg_negative_mf = ts_mean(negative_mf, period)
            
            # Calculate money flow ratio and MFI
            money_flow_ratio = divide(avg_positive_mf, avg_negative_mf)
            mfi = 100 - divide(100, add(money_flow_ratio, 1))
            
            df[f"mfi_{period}"] = mfi
        
        return df[_names]
