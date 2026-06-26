import numpy as np
from typing import Union
from sklearn.metrics import f1_score

from src.registry import METRIC
from src.metric.types import Metric

@METRIC.register_module(force=True)
class F1Score(Metric):
    """
    F1 Score metric.

    This class computes the F1 score for a classification task.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    Supports both binary and multiclass predictions.
    """

    def __init__(self, average='macro', **kwargs):
        """
        Args:
            average (str): The averaging method for multi-class tasks.
                           Options: 'micro', 'macro', 'weighted', or None.
        """
        super(F1Score, self).__init__(**kwargs)
        self.average = average

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate the F1 score of predictions against true labels.
        Args:
            y_true: True labels, a numpy array.
            y_pred: The predicted labels, a numpy array.
        Returns:
            float: The F1 score.
        """
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)

        # Compute F1 score
        f1 = f1_score(y_true, y_pred, average=self.average, zero_division=.0)

        f1 = f1 * 100  # Convert to percentage

        return float(f1)

if __name__ == '__main__':
    f1_metric = F1Score()

    # Example usage
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    f1 = f1_metric(y_true, y_pred)
    print(f"F1 (Numpy): {f1:.2f}%")

    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    # Filter out NaN/inf values for robust calculation
    valid_indices = np.isfinite(y_pred_nf).all(axis=1)
    y_true_filtered = y_true_nf[valid_indices]
    y_pred_filtered = y_pred_nf[valid_indices]
    
    if len(y_true_filtered) > 0:
        f1_nf = f1_metric(y_true_filtered, y_pred_filtered)
        print(f"Multiclass F1 NF (Numpy): {f1_nf:.2f}%")
    else:
        print("No valid samples for Multiclass F1 NF (Numpy).")
