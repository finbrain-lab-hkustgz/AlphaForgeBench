import pandas as pd
import numpy as np
from pydantic import Field
from typing import List

from src.operator import divide, min as min_op
from src.factor.types import Factor

EPS = 1e-12

class KLOW(Factor):
    """K-Low price factor."""
    
    type: str = Field(default="klow", description="The type of the factor")
    expression: str = Field(default="klow = (min(open, close) - low) / open, klow2 = (min(open, close) - low) / (high - low)", description="The expression of the factor")
    description: str = Field(default="K-Low price factor: measures the lower shadow of the candle", description="The description of the factor")
    names: List[str] = Field(default=["klow", "klow2"], description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = ["klow", "klow2"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the K-Low price factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
        
        Returns:
            pd.DataFrame: The DataFrame containing the KLOW indicators
        """
        # min_oc = min(open, close)
        min_oc = min_op(df["open"], df["close"])
        
        # klow = (min(open, close) - low) / open
        df["klow"] = divide(min_oc - df["low"], df["open"] + EPS)
        
        # klow2 = (min(open, close) - low) / (high - low)
        high_low_diff = df["high"] - df["low"]
        df["klow2"] = divide(min_oc - df["low"], high_low_diff + EPS)
        
        return df[self.names]




