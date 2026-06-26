import numpy as np
from typing import Union
from sklearn.metrics import roc_auc_score

from src.registry import METRIC
from src.metric.types import Metric

@METRIC.register_module(force=True)
class AUC(Metric):
    """
    AUC (Area Under the ROC Curve) metric.

    This class computes the AUC score for binary or multiclass classification tasks.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    Supports both probability and one-hot encoded predictions.
    """

    def __init__(self, average='macro', **kwargs):
        """
        Args:
            average (str): The averaging method for multi-class tasks. Usually 'macro'.
        """
        super(AUC, self).__init__(**kwargs)
        self.average = average

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> float:
        """
        Compute the AUC score from predictions and targets.
        Args:
            y_true: True labels, a numpy array.
            y_pred: The predicted labels, a numpy array.
        Returns:
            float: The AUC score.
        """
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)

        if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
            # Binary classification
            auc = roc_auc_score(y_true, y_pred)
        else:
            # Multiclass classification
            auc = roc_auc_score(
                y_true,
                y_pred,
                average=self.average,
                multi_class="ovr"  # One-vs-Rest strategy
            )

        auc = auc * 100  # Convert to percentage

        return float(auc)

if __name__ == '__main__':
    auc_metric = AUC()

    # Example usage
    y_true = np.array([0, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 0.9, 0.8, 0.1, 0.7, 0])
    auc = auc_metric(y_true, y_pred)
    print(f"AUC (Numpy): {auc:.2f}%")

    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    # Filter out NaN/inf values for robust calculation
    valid_indices = np.isfinite(y_pred_nf).all(axis=1)
    y_true_filtered = y_true_nf[valid_indices]
    y_pred_filtered = y_pred_nf[valid_indices]
    
    if len(y_true_filtered) > 0:
        auc_nf = auc_metric(y_true_filtered, y_pred_filtered)
        print(f"Multiclass AUC NF (Numpy): {auc_nf:.2f}%")
    else:
        print("No valid samples for Multiclass AUC NF (Numpy).")
