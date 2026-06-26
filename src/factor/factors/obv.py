import pandas as pd
import numpy as np
from pydantic import Field
from typing import List

from src.operator import delta, sign, multiply
from src.factor.types import Factor

class OBV(Factor):
    """On-Balance Volume (OBV) factor."""
    
    type: str = Field(default="obv", description="The type of the factor")
    expression: str = Field(default="obv = cumsum(volume * sign(close - delay(close, 1)))", description="The expression of the factor")
    description: str = Field(default="On-Balance Volume (OBV) indicator", description="The description of the factor")
    names: List[str] = Field(default=["obv"], description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = ["obv"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the On-Balance Volume (OBV) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price and volume data
        
        Returns:
            pd.DataFrame: The DataFrame containing the OBV indicator
        """
        
        # Calculate price change direction
        delta_close = delta(df["close"], 1)
        
        # Calculate sign of price change
        price_sign = sign(delta_close)
        
        # Calculate OBV: cumulative sum of volume * sign(price_change)
        obv_volume = multiply(df["volume"], price_sign)
        obv = obv_volume.cumsum()
        
        df["obv"] = obv
        
        return df[self.names]
