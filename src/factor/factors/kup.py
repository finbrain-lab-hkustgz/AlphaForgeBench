import pandas as pd
import numpy as np
from pydantic import Field
from typing import List

from src.operator import divide, max as max_op
from src.factor.types import Factor

EPS = 1e-12

class KUP(Factor):
    """K-Up price factor."""
    
    type: str = Field(default="kup", description="The type of the factor")
    expression: str = Field(default="kup = (high - max(open, close)) / open, kup2 = (high - max(open, close)) / (high - low)", description="The expression of the factor")
    description: str = Field(default="K-Up price factor: measures the upper shadow of the candle", description="The description of the factor")
    names: List[str] = Field(default=["kup", "kup2"], description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = ["kup", "kup2"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the K-Up price factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
        
        Returns:
            pd.DataFrame: The DataFrame containing the KUP indicators
        """
        # max_oc = max(open, close)
        max_oc = max_op(df["open"], df["close"])
        
        # kup = (high - max(open, close)) / open
        df["kup"] = divide(df["high"] - max_oc, df["open"] + EPS)
        
        # kup2 = (high - max(open, close)) / (high - low)
        high_low_diff = df["high"] - df["low"]
        df["kup2"] = divide(df["high"] - max_oc, high_low_diff + EPS)
        
        return df[self.names]




