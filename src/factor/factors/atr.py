import pandas as pd
from pydantic import Field
from typing import List, Optional

from src.operator import abs, delay, max, ts_mean
from src.factor.types import Factor

class ATR(Factor):
    """Average True Range (ATR) factor."""
    
    type: str = Field(default="atr", description="The type of the factor")
    expression: str = Field(default="atr = ts_mean(max(high - low, abs(high - delay(close, 1)), abs(low - delay(close, 1))), period)", description="The expression of the factor")
    description: str = Field(default="Average True Range (ATR) indicator", description="The description of the factor")
    names: List[str] = Field(default=["atr_14"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [14], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.names = [f"atr_{period}" for period in periods]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Average True Range (ATR) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the ATR indicator
        """
        _periods = periods if periods is not None else self.periods
        _names = [f"atr_{period}" for period in _periods]
        
        for period in _periods:
            df[f"atr_{period}"] = ts_mean(max(df["high"] - df["low"], abs(df["high"] - delay(df["close"], 1)), abs(df["low"] - delay(df["close"], 1))), period)
        
        return df[_names]
