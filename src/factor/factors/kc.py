import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import delay, max as max_op, abs as abs_op, add, multiply, ts_mean
from src.factor.types import Factor

class KC(Factor):
    """Keltner Channels (KC) factor."""
    
    type: str = Field(default="kc", description="The type of the factor")
    expression: str = Field(default="kc_upper = ts_mean(close, period) + ts_mean(tr, period) * nbdev, kc_middle = ts_mean(close, period), kc_lower = ts_mean(close, period) - ts_mean(tr, period) * nbdev", description="The expression of the factor")
    description: str = Field(default="Keltner Channels (KC) indicator", description="The description of the factor")
    names: List[str] = Field(default=["kc_upper_20", "kc_middle_20", "kc_lower_20"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [20], nbdev: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.nbdev = nbdev
        self.names = []
        for period in periods:
            self.names.extend([f"kc_upper_{period}", f"kc_middle_{period}", f"kc_lower_{period}"])

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Keltner Channels (KC) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Dynamic periods override. If None, uses self.periods.
        
        Returns:
            pd.DataFrame: The DataFrame containing the KC indicator (upper, middle, lower bands)
        """
        _periods = periods if periods is not None else self.periods
        _names = []
        for period in _periods:
            _names.extend([f"kc_upper_{period}", f"kc_middle_{period}", f"kc_lower_{period}"])
        
        close = df['close']
        high = df['high'] if 'high' in df.columns else close
        low = df['low'] if 'low' in df.columns else close

        # Calculate True Range using operators
        prev_close = delay(close, 1)
        tr1 = add(high, multiply(low, -1))
        tr2 = abs_op(add(high, multiply(prev_close, -1)))
        tr3 = abs_op(add(low, multiply(prev_close, -1)))
        tr = max_op(tr1, tr2, tr3)
            
        for period in _periods:
            # Middle band is SMA of close
            middle = ts_mean(close, period)
            
            # ATR using SMA
            atr = ts_mean(tr, period)
            
            df[f"kc_middle_{period}"] = middle
            df[f"kc_upper_{period}"] = add(middle, multiply(atr, self.nbdev))
            df[f"kc_lower_{period}"] = add(middle, multiply(atr, -self.nbdev))
        
        return df[_names]
