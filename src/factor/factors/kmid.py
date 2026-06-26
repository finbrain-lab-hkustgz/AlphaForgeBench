import pandas as pd
import numpy as np
from pydantic import Field
from typing import List

from src.operator import divide, add
from src.factor.types import Factor

EPS = 1e-12

class KMID(Factor):
    """K-Mid price factor."""
    
    type: str = Field(default="kmid", description="The type of the factor")
    expression: str = Field(default="kmid = (close - open) / close, kmid2 = (close - open) / (high - low)", description="The expression of the factor")
    description: str = Field(default="K-Mid price factor: measures the relative position of close price within the candle", description="The description of the factor")
    names: List[str] = Field(default=["kmid", "kmid2"], description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = ["kmid", "kmid2"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the K-Mid price factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
        
        Returns:
            pd.DataFrame: The DataFrame containing the KMID indicators
        """
        # kmid = (close - open) / close
        df["kmid"] = divide(df["close"] - df["open"], df["close"] + EPS)
        
        # kmid2 = (close - open) / (high - low)
        high_low_diff = df["high"] - df["low"]
        df["kmid2"] = divide(df["close"] - df["open"], high_low_diff + EPS)
        
        return df[self.names]




