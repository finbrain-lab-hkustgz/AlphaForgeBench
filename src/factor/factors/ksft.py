import pandas as pd
import numpy as np
from pydantic import Field
from typing import List

from src.operator import divide, multiply, add
from src.factor.types import Factor

EPS = 1e-12

class KSFT(Factor):
    """K-Shift price factor."""
    
    type: str = Field(default="ksft", description="The type of the factor")
    expression: str = Field(default="ksft = (2 * close - high - low) / open, ksft2 = (2 * close - high - low) / (high - low)", description="The expression of the factor")
    description: str = Field(default="K-Shift price factor: measures the shift of close price relative to the candle range", description="The description of the factor")
    names: List[str] = Field(default=["ksft", "ksft2"], description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = ["ksft", "ksft2"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the K-Shift price factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
        
        Returns:
            pd.DataFrame: The DataFrame containing the KSFT indicators
        """
        # ksft = (2 * close - high - low) / open
        numerator = multiply(df["close"], 2) - df["high"] - df["low"]
        df["ksft"] = divide(numerator, df["open"] + EPS)
        
        # ksft2 = (2 * close - high - low) / (high - low)
        high_low_diff = df["high"] - df["low"]
        df["ksft2"] = divide(numerator, high_low_diff + EPS)
        
        return df[self.names]




