from typing import List, Dict, Any
from pydantic import Field
import pandas as pd

from src.strategy.types import Strategy

class BuyAndHold(Strategy):
    """Buy and Hold trading strategy.
    
    This strategy generates trading signals as follows:
    - Buy signal (1): On the first call, buy and hold
    - Hold signal (0): After the first buy, always hold
    - Never sell (-1): This strategy never sells once bought
    """
    
    name: str = Field(default="buy_and_hold", description="The name of the strategy")
    description: str = Field(default="Buy and Hold trading strategy - buy once and hold forever", description="The description of the strategy")
    factor_names: List[str] = Field(default=[], description="The names of the factors (none required for this strategy)")
    
    def __init__(self, **kwargs):
        """
        Initialize Buy and Hold trading strategy.
        
        Args:
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(**kwargs)
        self._has_bought = False  # Track if we've already bought
    
    async def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal for buy and hold strategy.
        
        This strategy:
        - Buys on the first call (signal = 1)
        - Holds on all subsequent calls (signal = 0)
        - Never sells (signal never = -1)
        
        Args:
            df (pd.DataFrame): The input DataFrame with shape (d, features) containing:
                - d: Number of historical time steps (lookback window)
                - Price data: open, high, low, close, volume
                - No factors required for this strategy
        
        Returns:
            Dict[str, Any]: The dictionary containing the strategy outputs:
                - signal: Side of the order (buy or sell), -1 for sell, 1 for buy, 0 for hold
                - position: Quantity of the order (default 1.0, can be customized)
        """
        # Buy on first call, then always hold
        if not self._has_bought:
            signal = 1  # Buy signal on first call
            self._has_bought = True
        else:
            signal = 0  # Hold signal after first buy (never sell)
        
        order: Dict[str, Any] = {
            "signal": signal,
            "position": 1.0
        }
        
        return order

