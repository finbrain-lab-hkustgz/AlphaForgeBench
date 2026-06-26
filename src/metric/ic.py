import numpy as np
from typing import List, Optional, Any, Union

from src.registry import METRIC
from src.metric.types import Metric

@METRIC.register_module(force=True)
class IC(Metric):
    """Information Coefficient (IC) metric.

    This class computes the Information Coefficient (Pearson correlation coefficient)
    between true and predicted values.
    It handles NaN and infinite values by filtering them out.
    """
    def __init__(self, **kwargs):
        """
        Initialize the IC metric.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super(IC, self).__init__(**kwargs)

    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 mask: Optional[np.ndarray] = None
                 ) -> float:
        """Compute the Information Coefficient (Pearson correlation) between true and predicted values.

        Args:
            y_true (np.ndarray): True values (e.g., future returns).
            y_pred (np.ndarray): Predicted values (e.g., factor values).
            mask (Optional[np.ndarray]): Optional mask to apply to the true and predicted values.
        Returns:
            float: The computed Information Coefficient (Pearson correlation coefficient).
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if mask is not None:
            mask = np.asarray(mask, dtype=bool).ravel()
            keep = ~mask
        else:
            keep = np.ones_like(y_true, dtype=bool)

        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        keep &= finite

        if keep.sum() < 2:  # Not enough data to compute IC
            return 0.0

        y_true = y_true[keep]
        y_pred = y_pred[keep]

        # Calculate Pearson correlation coefficient
        # IC = Cov(y_true, y_pred) / (std(y_true) * std(y_pred))
        cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
        std_true = y_true.std()
        std_pred = y_pred.std()

        if std_true == 0 or std_pred == 0:
            return 0.0

        ic = cov / (std_true * std_pred)

        return float(ic)

if __name__ == '__main__':
    ic_metric = IC()

    # Example usage with NumPy arrays
    y_true_np = np.array([1, 2, 3, 4, 5])
    y_pred_np = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
    ic_score_np = ic_metric(y_true_np, y_pred_np)
    print(f"IC (Numpy): {ic_score_np:.4f}")

    # Example with perfect negative correlation
    y_true_np2 = np.array([1, 2, 3, 4, 5])
    y_pred_np2 = np.array([5, 4, 3, 2, 1])
    ic_score_np2 = ic_metric(y_true_np2, y_pred_np2)
    print(f"IC (Perfect negative correlation): {ic_score_np2:.4f}")

    # Example with NaN values
    y_true_np3 = np.array([1, 2, np.nan, 4, 5])
    y_pred_np3 = np.array([1.1, 2.2, 3.0, 4.1, 5.0])
    ic_score_np3 = ic_metric(y_true_np3, y_pred_np3)
    print(f"IC (With NaN): {ic_score_np3:.4f}")

