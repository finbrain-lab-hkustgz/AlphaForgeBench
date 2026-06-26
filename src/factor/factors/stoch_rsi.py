import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.factor.types import Factor
from src.operator import ts_mean, ts_max, ts_min, delta, if_else, lt, gt, multiply, add, divide

class StochRSI(Factor):
    """Stochastic RSI (StochRSI) factor.
    
    Measures RSI relative to its high/low range over a given period.
    Outputs the main %K line (fast) and the smoothed %D line (slow).
    
    Standard settings: RSI period 14, Stoch period 14, K smooth 3, D smooth 3.
    Range: [0, 100]
    """
    
    type: str = Field(default="stoch_rsi", description="The type of the factor")
    expression: str = Field(default="stoch_rsi_k = ts_mean((RSI - min_RSI)/(max_RSI - min_RSI), K), stoch_rsi_d = ts_mean(stoch_rsi_k, D)", description="The expression of the factor")
    description: str = Field(default="Stochastic RSI indicator", description="The description of the factor")
    names: List[str] = Field(default=["stoch_rsi_k_14_14_3", "stoch_rsi_d_14_14_3"], description="The returned names of the factors")
    
    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, k: int = 3, d: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k = k
        self.d = d
        self.names = [
            f"stoch_rsi_k_{rsi_period}_{stoch_period}_{k}",
            f"stoch_rsi_d_{rsi_period}_{stoch_period}_{k}_{d}"
        ]

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate the Stochastic RSI indicator using operators.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            periods (Optional[List[int]]): Ignored here. Uses init params.
        
        Returns:
            pd.DataFrame: The DataFrame containing %K and %D lines.
        """
        close = df['close']
        
        # Calculate RSI components
        diff = delta(close, 1)
        
        gain_mask = gt(diff, 0)
        gain = if_else(gain_mask, diff, 0)
        
        loss_mask = lt(diff, 0)
        loss = if_else(loss_mask, multiply(diff, -1), 0)
        
        # SMA-based RSI (using ts_mean instead of ewm for strict operator usage)
        avg_gain = ts_mean(gain, self.rsi_period)
        avg_loss = ts_mean(loss, self.rsi_period)
        
        # rs = avg_gain / avg_loss
        rs = divide(avg_gain, avg_loss)
        
        # rsi = 100 - (100 / (1 + rs))
        rsi = add(100, multiply(divide(100, add(1, rs)), -1))
        
        # Calculate StochRSI components
        lowest_rsi = ts_min(rsi, self.stoch_period)
        highest_rsi = ts_max(rsi, self.stoch_period)
        
        # range_rsi = highest_rsi - lowest_rsi
        range_rsi = add(highest_rsi, multiply(lowest_rsi, -1))
        
        # Prevent division by zero
        is_zero = lt(range_rsi, 1e-8)
        range_rsi_safe = if_else(is_zero, 1e-8, range_rsi)
        
        # stoch_rsi = 100 * (rsi - lowest_rsi) / range_rsi_safe
        stoch_rsi = multiply(divide(add(rsi, multiply(lowest_rsi, -1)), range_rsi_safe), 100)
        
        # Smooth %K and %D
        k_name = f"stoch_rsi_k_{self.rsi_period}_{self.stoch_period}_{self.k}"
        d_name = f"stoch_rsi_d_{self.rsi_period}_{self.stoch_period}_{self.k}_{self.d}"
        
        df[k_name] = ts_mean(stoch_rsi, self.k)
        df[d_name] = ts_mean(df[k_name], self.d)
        
        return df[[k_name, d_name]]
