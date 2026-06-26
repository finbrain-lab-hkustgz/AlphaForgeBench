import numpy as np
from typing import Union
from sklearn.metrics import accuracy_score

from src.registry import METRIC
from src.metric.types import Metric

@METRIC.register_module(force=True)
class Accuracy(Metric):
    """
    Accuracy metric.

    This class computes the accuracy of a classification task.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    Supports both binary and multiclass predictions.
    """

    def __init__(self, **kwargs):
        super(Accuracy, self).__init__(**kwargs)

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate the accuracy of predictions against true labels.
        
        Args:
            y_true: True labels as numpy array
            y_pred: Predicted labels as numpy array (can be probabilities or class indices)
            
        Returns:
            float: The accuracy score as a percentage (0-100)
        """
        # Convert to numpy arrays if needed
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Handle multiclass predictions (probabilities)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)

        # Remove NaN and infinite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            return 0.0

        # Compute accuracy
        acc = accuracy_score(y_true_clean, y_pred_clean)

        # Convert to percentage
        acc = acc * 100

        return float(acc)

if __name__ == '__main__':
    accuracy_metric = Accuracy()

    # Example usage
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    acc = accuracy_metric(y_true, y_pred)
    print(f"Accuracy: {acc:.2f}%")

    # Multiclass with probabilities
    y_true_multiclass = np.array([0, 1, 2])
    y_pred_multiclass = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
    acc_multiclass = accuracy_metric(y_true_multiclass, y_pred_multiclass)
    print(f"Multiclass Accuracy: {acc_multiclass:.2f}%")

    # Handle NaN and infinite values
    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    acc_nf = accuracy_metric(y_true_nf, y_pred_nf)
    print(f"Accuracy with NaN/Inf: {acc_nf:.2f}%")
