import numpy as np
from typing import List, Optional, Any, Union, Dict

from src.registry import METRIC
from src.calendar import calendar_manager
from src.metric.utils import clean_invalid_values
from src.metric.types import Metric

@METRIC.register_module(force=True)
class ARR(Metric):
    """Annualized Return Rate (ARR) metric.

    This class computes the Annualized Return Rate based on the returns of a financial asset.
    It handles NaN and infinite values by removing them from the returns array.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: Dict[str, Any] = None,
                 **kwargs
                 ):
        """
        Initialize the ARR metric.

        Args:
            level (str): The time level for which the ARR is computed. Default is "1day".
        """
        super(ARR, self).__init__(**kwargs)
        self.level = level
        self._symbol_info = symbol_info

    def __call__(self,
                 ret: np.ndarray
                 ) -> float:
        """Compute the ARR from returns.

        Args:
            ret (np.ndarray): Returns of the asset.
        Returns:
            float: The computed Annualized Return Rate.
        """
        # process nan and inf
        ret = clean_invalid_values(ret)

        num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                       level = self.level)

        arr = (np.prod(1 + ret) ** (num_periods / len(ret))) - 1

        return float(arr)

if __name__ == '__main__':
    method = ARR(
        level="1day",
        symbol_info={
            "symbol": "AAPL",
            "exchange": "New York Stock Exchange",
        }
    )

    ret = np.array([0.01, 0.02, -0.005, 0.03, np.nan, np.inf])

    res = method(ret)
    print("Result:", res)