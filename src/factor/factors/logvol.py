import pandas as pd
import numpy as np
from pydantic import Field
from typing import List

from src.operator import log
from src.factor.types import Factor

class LOGVOL(Factor):
    """Log Volume factor from Alpha158."""
    
    type: str = Field(default="logvol", description="The type of the factor")
    expression: str = Field(default="logvol = log(volume + 1)", description="The expression of the factor")
    description: str = Field(default="Logarithm of Volume indicator from Alpha158", description="The description of the factor")
    names: List[str] = Field(default=["logvol"], description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = ["logvol"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Log Volume factor.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            
        Returns:
            pd.DataFrame: The DataFrame containing the LOGVOL indicator
        """
        df["logvol"] = log(df["volume"] + 1)
        
        return df[self.names]

