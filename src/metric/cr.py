import numpy as np
from typing import Any, Union, Dict

from src.registry import METRIC
from src.metric.mdd import MDD
from src.metric.arr import ARR
from src.metric.types import Metric

@METRIC.register_module(force=True)
class CR(Metric):
    """Calmar Ratio (CR) metric.

    This class computes the Calmar Ratio, which is the ratio of the annualized return to the maximum drawdown.
    It handles NaN and infinite values by removing them from the returns array.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: Dict[str, Any] = None,
                 **kwargs
                 ):
        """
        Initialize the CR metric.

        Args:
            level (str): The time level for which the CR is computed. Default is "1day".
        """
        super(CR, self).__init__(**kwargs)

        self.level = level
        self._symbol_info = symbol_info

        self.arr = ARR(level=level, symbol_info=symbol_info)
        self.mdd = MDD(level=level, symbol_info=symbol_info)

    def __call__(self,
                 ret: np.ndarray
                 ) -> float:
        """Compute the CR from returns.

        Args:
            ret (np.ndarray): Returns of the asset.
        Returns:
            float: The computed Calmar Ratio.
        """
        arr = self.arr(ret)
        mdd = self.mdd(ret)

        mdd = np.abs(mdd)

        cr = arr / (mdd + 1e-12)

        return float(cr)

if __name__ == '__main__':
    method = CR(
        level="1day",
        symbol_info={
            "symbol": "AAPL",
            "exchange": "New York Stock Exchange",
        }
    )

    ret = np.array([0.01, 0.02, -0.005, 0.03, np.nan, np.inf])

    res = method(ret)
    print("Result:", res)