import numpy as np
from typing import List, Optional, Any, Union

from src.registry import METRIC
from src.metric.types import Metric

@METRIC.register_module(force=True)
class RANKIC(Metric):
    """Rank Information Coefficient (RankIC) metric.

    This class computes the Rank Information Coefficient between true and predicted values.
    It handles NaN and infinite values by replacing them with zero.
    """
    def __init__(self, **kwargs):
        """
        Initialize the RankIC metric.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super(RANKIC, self).__init__(**kwargs)

    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 mask: Optional[np.ndarray] = None
                 ) -> float:
        """Compute the Rank Information Coefficient between true and predicted values.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            mask (Optional[np.ndarray]): Optional mask to apply to the true and predicted values.
        Returns:
            float: The computed Rank Information Coefficient.
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

        if keep.sum() < 2: # Not enough data to compute RankIC
            return .0

        y_true = y_true[keep]
        y_pred = y_pred[keep]

        true_rank = y_true.argsort().argsort().astype(float)
        pred_rank = y_pred.argsort().argsort().astype(float)

        cov = np.mean(
            (true_rank - true_rank.mean()) *
            (pred_rank - pred_rank.mean())
        )
        std = true_rank.std() * pred_rank.std()

        rank_ic = cov / (std + 1e-12)

        return float(rank_ic)

if __name__ == '__main__':
    rankic_metric = RANKIC()

    # Example usage with NumPy arrays
    y_true_np = np.array([1, 2, 3, 4, 5])
    y_pred_np = np.array([5, 4, 3, 2, 1])
    rankic_score_np = rankic_metric(y_true_np, y_pred_np)
    print(f"RankIC (Numpy): {rankic_score_np:.4f}")