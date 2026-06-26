import numpy as np
from typing import List, Optional, Any, Union, Dict

from src.registry import METRIC
from src.calendar import calendar_manager
from src.metric.utils import clean_invalid_values
from src.metric.types import Metric

@METRIC.register_module(force=True)
class VOL(Metric):
    """Volatility metric.

    This class computes the volatility of a financial asset based on its returns.
    It handles NaN and infinite values by removing them from the returns array.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: Dict[str, Any] = None,
                 **kwargs
                 ):
        """
        Initialize the VOL metric.

        Args:
            level (str): The time level for which the volatility is computed. Default is "1day".
        """
        super(VOL, self).__init__(**kwargs)

        self.level = level
        self._symbol_info = symbol_info

    def __call__(self,
                 ret: np.ndarray
                 ) -> float:
        """Compute the volatility from returns.

        Args:
            ret (np.ndarray): Returns of the asset.
        Returns:
            float: The computed volatility.
        """
        # process nan and inf
        ret = clean_invalid_values(ret)

        num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                       level=self.level)

        vol = np.std(ret) * np.sqrt(num_periods)

        return float(vol)

if __name__ == '__main__':
    method = VOL(
        level="1day",
        symbol_info={
            "symbol": "AAPL",
            "exchange": "New York Stock Exchange",
        }
    )

    ret = np.array([0.01, 0.02, -0.005, 0.03, np.nan, np.inf])

    res = method(ret)
    print("Result:", res)