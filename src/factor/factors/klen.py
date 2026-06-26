import pandas as pd
import numpy as np
from pydantic import Field
from typing import List

from src.operator import divide
from src.factor.types import Factor

EPS = 1e-12

class KLEN(Factor):
    """K-Length price factor."""
    
    type: str = Field(default="klen", description="The type of the factor")
    expression: str = Field(default="klen = (high - low) / open", description="The expression of the factor")
    description: str = Field(default="K-Length price factor: measures the relative range of the candle", description="The description of the factor")
    names: List[str] = Field(default=["klen"], description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = ["klen"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the K-Length price factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
        
        Returns:
            pd.DataFrame: The DataFrame containing the KLEN indicator
        """
        # klen = (high - low) / open
        df["klen"] = divide(df["high"] - df["low"], df["open"] + EPS)
        
        return df[self.names]

