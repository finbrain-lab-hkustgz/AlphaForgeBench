import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import (
    delay,
    max as max_op,
    abs as abs_op,
    add,
    multiply,
    divide,
    ts_mean,
    ts_stddev,
    ts_max,
    ts_min,
    gt,
    lt,
    and_op,
)
from src.factor.types import Factor

class TTM_Squeeze(Factor):
    """TTM Squeeze Factor.
    
    Detects volatility compression (squeeze) when Bollinger Bands are inside Keltner Channels.
    Also computes a momentum histogram to indicate the direction of the breakout.
    
    Outputs:
      - squeeze_on_{period}: True (1) if in squeeze, False (0) otherwise.
      - squeeze_mom_{period}: Momentum proxy = ts_mean(close - avg(donchian_mid, sma), period).
    """
    
    type: str = Field(default="ttm_squeeze", description="The type of the factor")
    expression: str = Field(
        default="squeeze_on = BB_inside_KC, squeeze_mom = ts_mean(Close - avg(DonchianMid, SMA), period)",
        description="The expression of the factor",
    )
    description: str = Field(default="TTM Squeeze volatility and momentum indicator", description="The description of the factor")
    names: List[str] = Field(default=["squeeze_on_20", "squeeze_mom_20"], description="The returned names of the factors")
    
    def __init__(self, periods: List[int] = [20], bb_mult: float = 2.0, kc_mult: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
        self.bb_mult = bb_mult
        self.kc_mult = kc_mult
        self.names = []
        for period in periods:
            self.names.extend([f"squeeze_on_{period}", f"squeeze_mom_{period}"])

    async def __call__(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        _periods = periods if periods is not None else self.periods
        _names = []
        for period in _periods:
            _names.extend([f"squeeze_on_{period}", f"squeeze_mom_{period}"])
        
        close = df['close']
        high = df['high'] if 'high' in df.columns else close
        low = df['low'] if 'low' in df.columns else close
        
        # Calculate TR using operators
        prev_close = delay(close, 1)
        tr1 = add(high, multiply(low, -1))
        tr2 = abs_op(add(high, multiply(prev_close, -1)))
        tr3 = abs_op(add(low, multiply(prev_close, -1)))
        tr = max_op(tr1, tr2, tr3)

        for period in _periods:
            # BB using operators
            sma = ts_mean(close, period)
            std = ts_stddev(close, period)
            bb_upper = add(sma, multiply(std, self.bb_mult))
            bb_lower = add(sma, multiply(std, -self.bb_mult))
            
            # KC using operators
            kc_middle = sma
            atr = ts_mean(tr, period)
            kc_upper = add(kc_middle, multiply(atr, self.kc_mult))
            kc_lower = add(kc_middle, multiply(atr, -self.kc_mult))
            
            # Squeeze On: BB is completely inside KC -> bb_lower > kc_lower AND bb_upper < kc_upper
            cond1 = gt(bb_lower, kc_lower)
            cond2 = lt(bb_upper, kc_upper)
            squeeze_on = and_op(cond1, cond2)
            df[f"squeeze_on_{period}"] = squeeze_on.astype(int)
            
            # Momentum proxy: mean-reverting vs trending bias of price relative to midline
            highest_high = ts_max(high, period)
            lowest_low = ts_min(low, period)
            
            # Donchian Mid
            donchian_mid = divide(add(highest_high, lowest_low), 2.0)
            
            # Avg(DonchianMid, SMA)
            avg_mid_sma = divide(add(donchian_mid, sma), 2.0)
            
            # Val = Close - Avg(DonchianMid, SMA)
            val = add(close, multiply(avg_mid_sma, -1))
            df[f"squeeze_mom_{period}"] = ts_mean(val, period)
            
        return df[_names]
