import numpy as np
from typing import List, Optional, Any, Union, Dict

from src.registry import METRIC
from src.calendar import calendar_manager
from src.metric.utils import clean_invalid_values
from src.metric.dd import DD
from src.metric.types import Metric

@METRIC.register_module(force=True)
class SOR(Metric):
    """Sortino Ratio (SOR) metric.

    This class computes the Sortino Ratio based on the returns of a financial asset.
    It handles NaN and infinite values by removing them from the returns array.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: Dict[str, Any] = None,
                 risk_free_rate: float = 0.0,
                 **kwargs
                 ):
        """
        Initialize the SOR metric.

        Args:
            level (str): The time level for which the SOR is computed. Default is "1day".
            risk_free_rate (float): The risk-free rate to use in the Sortino Ratio calculation. Default is 0.0.
        """
        super(SOR, self).__init__(**kwargs)

        self.level = level
        self._symbol_info = symbol_info
        self.risk_free_rate = risk_free_rate

        self.dd = DD(level=level, symbol_info=symbol_info)

    def __call__(self,
                 ret: np.ndarray
                 ) -> float:
        """Compute the SOR from returns.

        Args:
            ret (np.ndarray): Returns of the asset.
        Returns:
            float: The computed Sortino Ratio.
        """
        # process nan and inf
        ret = clean_invalid_values(ret)

        num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                       level=self.level)

        rf_period = self.risk_free_rate / num_periods
        excess_mean_period = ret.mean() - rf_period
        mu_ann = excess_mean_period * num_periods

        sigma_down_ann = self.dd(ret)

        sor = mu_ann / (sigma_down_ann + 1e-12)

        return float(sor)

if __name__ == '__main__':
    method = SOR(
        level="1day",
        symbol_info={
            "symbol": "AAPL",
            "exchange": "New York Stock Exchange",
        }
    )

    ret = np.array([0.01, 0.02, -0.005, 0.03, np.nan, np.inf])

    res = method(ret)
    print("Result:", res)