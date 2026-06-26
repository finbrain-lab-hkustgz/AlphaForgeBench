import warnings

import pandas as pd
import numpy as np
from pydantic import Field
from typing import List, Optional

from src.factor.types import Factor


class HurstExponent(Factor):
    """Hurst Exponent (approximation) for regime detection.

    Hurst exponent interpretation:
      - H < 0.5: anti-persistent / mean-reverting tendency
      - H > 0.5: persistent / trending tendency

    This implementation uses a fast, vectorized approximation based on the
    scaling relationship of lagged differences:
      std(close(t) - close(t-lag)) ~ lag^H
    For each timestamp, we estimate H as the slope of log(std) vs log(lag)
    over a small set of lags.

    Outputs: hurst_{window}
    """

    type: str = Field(default="hurst", description="The type of the factor")
    expression: str = Field(
        default="hurst = slope(log(std(close-close.shift(lag))), log(lag))",
        description="The expression of the factor",
    )
    description: str = Field(
        default="Hurst Exponent (approx) — mean reversion vs trend regime indicator",
        description="The description of the factor",
    )
    names: List[str] = Field(
        default=["hurst_120"],
        description="The returned names of the factors",
    )

    def __init__(self, windows: List[int] = [120], lags: List[int] = [2, 4, 8, 16], **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.lags = lags
        self.names = [f"hurst_{w}" for w in windows]

    async def __call__(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        _windows = windows if windows is not None else self.windows
        _lags = lags if lags is not None else self.lags
        _lags = [int(x) for x in _lags if int(x) >= 2]
        if len(_lags) < 2:
            raise ValueError("hurst factor requires at least 2 lags >= 2")

        _names = [f"hurst_{w}" for w in _windows]

        close = df["close"].astype(float)

        x = np.log(np.array(_lags, dtype=float))
        x_mean = float(x.mean())
        var_x = float(((x - x_mean) ** 2).mean())
        if var_x <= 0:
            var_x = 1e-12

        for w in _windows:
            std_cols = []
            for lag in _lags:
                d = close - close.shift(lag)
                std_cols.append(d.rolling(w).std())
            std_df = pd.concat(std_cols, axis=1)
            std_df.columns = [f"std_lag_{lag}" for lag in _lags]

            y = np.log(std_df.replace(0, np.nan))
            yv = y.to_numpy(dtype=float)
            # 前 w 根 bar 数据不足时 yv 全 NaN，np.nanmean 会触发 Mean of empty slice，此处抑制
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y_mean = np.nanmean(yv, axis=1, keepdims=True)
                cov = np.nanmean((yv - y_mean) * (x - x_mean), axis=1)
            hurst = cov / var_x

            hurst_s = pd.Series(hurst, index=df.index).clip(lower=0.0, upper=1.0)
            df[f"hurst_{w}"] = hurst_s

        return df[_names]

