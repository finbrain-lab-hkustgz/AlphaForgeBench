import numpy as np
from typing import Union
from sklearn.metrics import precision_score

from src.registry import METRIC
from src.metric.types import Metric

@METRIC.register_module(force=True)
class Precision(Metric):
    """Precision metric.

    This class computes the precision of a binary classification task.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    """

    def __init__(self,
                 average = 'macro',
                 **kwargs):
        super(Precision, self).__init__(**kwargs)
        self.average = average

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate the precision of predictions against true labels.
        Args:
            y_true: True labels, a numpy array.
            y_pred: The predicted labels, a numpy array.
        Returns:
            float: The precision score.
        """
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)

        # Compute precision
        pre = precision_score(y_true, y_pred, average=self.average)

        pre = pre * 100  # Convert to percentage

        return float(pre)

if __name__ == '__main__':
    precision_metric = Precision()

    # Example usage
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    pre = precision_metric(y_true, y_pred)
    print(f"Precision (Numpy): {pre:.2f}%")

    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    # Filter out NaN/inf values for robust calculation
    valid_indices = np.isfinite(y_pred_nf).all(axis=1)
    y_true_filtered = y_true_nf[valid_indices]
    y_pred_filtered = y_pred_nf[valid_indices]
    
    if len(y_true_filtered) > 0:
        pre_nf = precision_metric(y_true_filtered, y_pred_filtered)
        print(f"Multiclass Precision NF (Numpy): {pre_nf:.2f}%")
    else:
        print("No valid samples for Multiclass Precision NF (Numpy).")
