import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.operator import ts_mean, delta
from src.factor.types import Factor

class MACD(Factor):
    """Moving Average Convergence Divergence (MACD) factor."""
    
    type: str = Field(default="macd", description="The type of the factor")
    expression: str = Field(default="macd = ema(close, fastperiod) - ema(close, slowperiod), macd_signal = ema(ema(close, fastperiod) - ema(close, slowperiod), signalperiod), macd_hist = (ema(close, fastperiod) - ema(close, slowperiod)) - ema(ema(close, fastperiod) - ema(close, slowperiod), signalperiod)", description="The expression of the factor")
    description: str = Field(default="Moving Average Convergence Divergence (MACD) indicator", description="The description of the factor")
    names: List[str] = Field(default=["macd", "macd_signal", "macd_hist"], description="The returned names of the factors")
    
    def __init__(self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
        self.names = ["macd", "macd_signal", "macd_hist"]

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        multiplier = 2.0 / (period + 1)
        ema = pd.Series(index=series.index, dtype=float)
        ema.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            ema.iloc[i] = (series.iloc[i] - ema.iloc[i-1]) * multiplier + ema.iloc[i-1]
        
        return ema

    async def __call__(
        self,
        df: pd.DataFrame,
        fastperiod: Optional[int] = None,
        slowperiod: Optional[int] = None,
        signalperiod: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD) indicator.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the price data
            fastperiod (Optional[int]): Dynamic fast period override. If None, uses self.fastperiod.
            slowperiod (Optional[int]): Dynamic slow period override. If None, uses self.slowperiod.
            signalperiod (Optional[int]): Dynamic signal period override. If None, uses self.signalperiod.
        
        Returns:
            pd.DataFrame: The DataFrame containing the MACD indicator (macd, macd_signal, macd_hist)
        """
        _fast = fastperiod if fastperiod is not None else self.fastperiod
        _slow = slowperiod if slowperiod is not None else self.slowperiod
        _signal = signalperiod if signalperiod is not None else self.signalperiod
        
        # Calculate fast and slow EMA
        ema_fast = self._ema(df["close"], _fast)
        ema_slow = self._ema(df["close"], _slow)
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_signal = self._ema(macd, _signal)
        
        # Calculate histogram
        macd_hist = macd - macd_signal
        
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        
        return df[self.names]
