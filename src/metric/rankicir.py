import numpy as np
from typing import List, Optional, Any, Union

from src.registry import METRIC
from src.metric.utils import fill_invalid_values
from src.metric.types import Metric

@METRIC.register_module(force=True)
class RANKICIR(Metric):
    """Rank Information Coefficient Information Ratio (RankICIR) metric.

    This class computes the Rank Information Coefficient between true and predicted values,
    and then calculates the Cumulative Information Ratio.
    It handles NaN and infinite values by replacing them with zero.
    """
    def __init__(self, **kwargs):
        """
        Initialize the RankICIR metric.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super(RANKICIR, self).__init__(**kwargs)

    def __call__(self,
                 rankic_values: np.ndarray,
                 ) -> float:
        """Compute the Rank Information Coefficient Information Ratio.
        Args:
            rankic_values (np.ndarray): RankIC values.
        Returns:
            float: The computed Cumulative Information Ratio.
        """
        rankic_values = np.asarray(rankic_values).ravel()
        rankic_values = fill_invalid_values(rankic_values)

        randkicir = np.mean(rankic_values) / (np.std(rankic_values) + 1e-12)

        return float(randkicir)

if __name__ == '__main__':
    # Example usage
    rankicir_metric = RANKICIR()
    rankic_values = np.random.rand(100)  # Example RankIC values
    result = rankicir_metric(rankic_values)
    print(f"RankICIR: {result}")
