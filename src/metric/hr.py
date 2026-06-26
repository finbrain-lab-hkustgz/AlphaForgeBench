import numpy as np
from typing import Union

from src.registry import METRIC
from src.metric.types import Metric

@METRIC.register_module(force=True)
class HitRatio(Metric):
    """Hit Ratio metric.

    This class computes the hit ratio of a price prediction task.
    It measures the proportion of times the predicted price direction matches the actual price direction.
    """

    def __init__(self, **kwargs):
        super(HitRatio, self).__init__(**kwargs)

    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray
                 ) -> float:
        """Compute the hit ratio from predictions and targets."""
        # Remove nan/inf values (must do before diff, so both seq length match)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Not enough samples
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0

        # Calculate actual and predicted price changes (difference with previous)
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)

        # Calculate sign (1: up, -1: down, 0: flat)
        true_sign = np.sign(true_diff)
        pred_sign = np.sign(pred_diff)

        # Calculate hit (direction correct)
        hits = (true_sign == pred_sign)
        hit_ratio_score = np.mean(hits)

        hit_ratio_score = hit_ratio_score * 100  # Convert to percentage

        return float(hit_ratio_score)

if __name__ == '__main__':
    hit_ratio_metric = HitRatio()

    # Example usage
    y_true = np.array([100, 102, 101, 105, 107])
    y_pred = np.array([100, 101, 103, 104, 108])
    hit_ratio = hit_ratio_metric(y_true, y_pred)
    print(f"Hit Ratio: {hit_ratio}")