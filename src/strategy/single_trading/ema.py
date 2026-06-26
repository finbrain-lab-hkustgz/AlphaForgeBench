from typing import List, Dict, Any
from pydantic import Field
import pandas as pd

from src.strategy.types import Strategy

class EMA(Strategy):
    """Exponential Moving Average trading strategy.
    
    This strategy generates trading signals based on EMA crossover:
    - Buy side (1): When short EMA crosses above long EMA (golden cross)
    - Sell side (-1): When short EMA crosses below long EMA (death cross)
    - Hold side (0): Otherwise
    """
    
    name: str = Field(default="ema", description="The name of the strategy")
    description: str = Field(default="Exponential Moving Average trading strategy based on EMA crossover", description="The description of the strategy")
    factor_names: List[str] = Field(default=["ema"], description="The names of the factors")
    
    # Customizable parameters
    periods: List[int] = Field(default=[20, 50], description="The periods of the EMA")
    
    def __init__(self, periods: List[int] = [20, 50], **kwargs):
        """
        Initialize EMA trading strategy.
        
        Args:
            periods: List of EMA periods. The first period is short EMA, the second is long EMA.
                     Default is [20, 50] (20-day short EMA, 50-day long EMA).
        """
        super().__init__(**kwargs)
        if len(periods) < 2:
            raise ValueError("EMA strategy requires at least 2 periods (short and long EMA)")
        self.periods = periods
        self.factor_names = [f"ema_{period}" for period in periods]
        
    async def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal based on current day's EMA position.
        
        Args:
            df (pd.DataFrame): The input DataFrame with shape (d, features) containing:
                - d: Number of historical time steps (lookback window)
                - Price data: open, high, low, close, volume
                - EMA factors: ema_{short_period}, ema_{long_period} (must be pre-calculated)
                Strategy only uses the last row (current day) for decision making.
        
        Returns:
            Dict[str, Any]: The dictionary containing the strategy outputs:
                - signal: Side of the order (buy or sell), -1 for sell, 1 for buy, 0 for hold
                - position: Quantity of the order (default 1.0, can be customized)
        """
        # Check if required EMA columns exist
        short_ema_name = f"ema_{self.periods[0]}" # short EMA column name
        long_ema_name = f"ema_{self.periods[1]}" # long EMA column name
        
        if short_ema_name not in df.columns:
            raise ValueError(f"Required column '{short_ema_name}' not found in DataFrame. Please calculate EMA factor first.")
        if long_ema_name not in df.columns:
            raise ValueError(f"Required column '{long_ema_name}' not found in DataFrame. Please calculate EMA factor first.")
        
        # Get the last row (current day) for decision making
        current_row = df.iloc[-1]
        short_ema_current = current_row[short_ema_name]
        long_ema_current = current_row[long_ema_name]
        
        # Generate signal based on current day's EMA position
        # Buy signal: short EMA > long EMA (bullish)
        # Sell signal: short EMA < long EMA (bearish)
        # Hold signal: short EMA == long EMA (neutral)
        if pd.isna(short_ema_current) or pd.isna(long_ema_current):
            signal = 0  # Hold if data is missing
        elif short_ema_current > long_ema_current:
            signal = 1  # Buy signal
        elif short_ema_current < long_ema_current:
            signal = -1  # Sell signal
        else:
            signal = 0  # Hold signal
        
        order: Dict[str, Any] = {
            "signal": signal,
            "position": 1.0
        }
        
        return order