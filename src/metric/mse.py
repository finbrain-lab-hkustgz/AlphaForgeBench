import numpy as np
from typing import Optional

from src.registry import METRIC
from src.metric.utils import fill_invalid_values
from src.metric.types import Metric

@METRIC.register_module(force=True)
class MSE(Metric):
    """Mean Squared Error (MSE) metric.

    This class computes the Mean Squared Error between true and predicted values.
    It handles NaN and infinite values by removing them from the returns array.
    """
    def __init__(self, **kwargs):
        """
        Initialize the MSE metric.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super(MSE, self).__init__(**kwargs)

    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 mask: Optional[np.ndarray] = None
                 ) -> float:
        """Compute the MSE between true and predicted values.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            mask (Optional[np.ndarray]): Optional mask to apply to the true and predicted values.
        Returns:
            float: The computed Mean Squared Error.
        """
        # process nan and inf
        y_true = fill_invalid_values(y_true)
        y_pred = fill_invalid_values(y_pred)

        if mask is not None:
            y_true = y_true * (1.0 - mask)
            y_pred = y_pred * (1.0 - mask)

        mse = np.mean((y_true - y_pred) ** 2)

        return float(mse)

if __name__ == '__main__':
    mse_metric = MSE()
    y_true = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.0, np.inf])
    mse_score = mse_metric(y_true, y_pred)
    print(f"MSE Score: {mse_score}")
