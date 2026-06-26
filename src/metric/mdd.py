import numpy as np
from typing import Any, Union

from src.registry import METRIC
from src.metric.utils import clean_invalid_values
from src.metric.types import Metric

@METRIC.register_module(force=True)
class MDD(Metric):
    """Maximum Drawdown (MDD) metric.

    This class computes the Maximum Drawdown based on the returns of a financial asset.
    It handles NaN and infinite values by replacing them with zero.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: dict = None,
                 **kwargs
                 ):
        """
        Initialize the MDD metric.
        """
        super(MDD, self).__init__(**kwargs)
        self.level = level
        self._symbol_info = symbol_info

    def __call__(self,
                 ret: np.ndarray
                 ) -> float:
        """Compute the MDD from returns.

        Args:
            ret (np.ndarray): Returns of the asset.
        Returns:
            float: The computed Maximum Drawdown.
        """
        # process nan and inf
        ret = clean_invalid_values(ret)

        cumulative_returns = np.cumprod(1 + ret)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-12)

        mdd = np.max(drawdown)

        return float(mdd)

if __name__ == '__main__':
    method = MDD(
        level="1day",
        symbol_info={
            "symbol": "AAPL",
            "exchange": "New York Stock Exchange",
        }
    )

    ret = np.array([0.01, 0.02, -0.005, 0.03, np.nan, np.inf])

    res = method(ret)
    print("Result:", res)