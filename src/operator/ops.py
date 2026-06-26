import numpy as np
import pandas as pd
from typing import Union

EPS = 1e-10

OPERATORS_DESCRIPTION = """
There are 53 basic operators in total.

## Arithmetic Operators
- abs(x: pd.Series) -> pd.Series: Absolute value of x.
- add(x: pd.Series, y: Union[pd.Series, float], filter: bool = False) -> pd.Series: Add all inputs (at least 2 inputs required). If filter = true, filter all input NaN to 0 before adding.
- divide(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: x / y
- log(x: pd.Series) -> pd.Series: Natural logarithm of x.
- max(x: Union[pd.Series, float], y: Union[pd.Series, float], *args: Union[pd.Series, float]) -> pd.Series: Element-wise maximum of 2 or more inputs.
- min(x: Union[pd.Series, float], y: Union[pd.Series, float], *args: Union[pd.Series, float]) -> pd.Series: Element-wise minimum of 2 or more inputs.
- multiply(x: pd.Series, y: Union[pd.Series, float], filter: bool = False) -> pd.Series: Multiply all inputs. At least 2 inputs are required. Filter sets the NaN values to 1.
- power(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: x raised to the power of y
- reverse(x: pd.Series) -> pd.Series: -x
- sign(x: pd.Series) -> pd.Series: if input > 0, return 1; if input < 0, return -1; if input = 0, return 0; if input = NaN, return NaN.
- signed_power(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: x raised to the power of y such that final result preserves sign of x.
- sqrt(x: pd.Series) -> pd.Series: Square root of x.

## Logical Operators
- and_op(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: Logical AND operator, returns true if both operands are true and returns false otherwise.
- if_else(x: pd.Series, y: pd.Series, z: Union[pd.Series, float]) -> pd.Series: If x is true then return y else return z.
- lt(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: If x < y return true, else return false.
- le(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: Returns true if x <= y, return false otherwise.
- eq(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: Returns true if both inputs are same and returns false otherwise.
- gt(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: Logic comparison operators to compares two inputs.
- ge(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: Returns true if x >= y, return false otherwise.
- ne(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series: Returns true if both inputs are NOT the same and returns false otherwise.
- is_nan(x: pd.Series) -> pd.Series: If (x == NaN) return 1 else return 0.

## Time Series Operators
- delay(x: pd.Series, period: int) -> pd.Series: Shift the series backward by the specified number of periods.
- ts_delay(x: pd.Series, period: int) -> pd.Series: Time series delay: shift the series backward by the specified number of periods.
- delta(x: pd.Series, period: int = 1) -> pd.Series: Difference between current value and value n periods ago.
- ts_delta(x: pd.Series, period: int) -> pd.Series: Time series delta: difference between current value and value n periods ago.
- ts_rank(x: pd.Series, period: int) -> pd.Series: Time series rank: rank of current value within the rolling window.
- ts_sum(x: pd.Series, period: int) -> pd.Series: Rolling sum over the specified number of periods.
- ts_mean(x: pd.Series, period: int) -> pd.Series: Rolling mean over the specified number of periods.
- ts_max(x: pd.Series, period: int) -> pd.Series: Rolling maximum over the specified number of periods.
- ts_min(x: pd.Series, period: int) -> pd.Series: Rolling minimum over the specified number of periods.
- ts_stddev(x: pd.Series, period: int) -> pd.Series: Rolling standard deviation over the specified number of periods.
- ts_std_dev(x: pd.Series, period: int) -> pd.Series: Rolling standard deviation over the specified number of periods.
- ts_corr(x: pd.Series, y: pd.Series, period: int) -> pd.Series: Returns correlation of x and y for the past d days.
- ts_covariance(x: pd.Series, y: pd.Series, period: int) -> pd.Series: Rolling covariance between two series over the specified number of periods.
- ts_argmax(x: pd.Series, period: int) -> pd.Series: Returns the relative index of the max value in the time series for the past d days.
- ts_arg_min(x: pd.Series, period: int) -> pd.Series: Returns the relative index of the min value in the time series for the past d days.
- ts_product(x: pd.Series, period: int) -> pd.Series: Rolling product over the specified number of periods.
- ts_zscore(x: pd.Series, period: int) -> pd.Series: Time series z-score: (x - ts_mean(x, d)) / ts_std_dev(x, d).
- ts_av_diff(x: pd.Series, period: int) -> pd.Series: Returns x - ts_mean(x, d), but deals with NaNs carefully. That is NaNs are ignored during mean computation.
- days_from_last_change(x: pd.Series) -> pd.Series: Amount of days since last change of x.
- hump(x: pd.Series, hump: float = 0.01) -> pd.Series: Limits amount and magnitude of changes in input (thus reducing turnover).
- kth_element(x: pd.Series, d: int, k: int) -> pd.Series: Returns K-th valid value of input by looking through lookback days. This operator can be used to backfill missing data if k=1.
- last_diff_value(x: pd.Series, d: int) -> pd.Series: Returns last x value not equal to current x value from last d days.

## Vector Operators
- rank(x: pd.Series) -> pd.Series: Cross-sectional rank of the input series (percentile rank).
- zscore(x: pd.Series) -> pd.Series: Z-score normalization: (x - mean(x)) / std(x).
- quantile(x: pd.Series, percentage: float) -> pd.Series: Quantile of the input series.
- normalize(x: pd.Series) -> pd.Series: Normalize the series to [0, 1] range.
"""


# ============ Arithmetic Operators ============

def abs(x: pd.Series) -> pd.Series:
    """
    Absolute value of x.
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The absolute value of the input series
    """
    return x.abs()


def add(x: pd.Series, y: Union[pd.Series, float], filter: bool = False) -> pd.Series:
    """
    Add all inputs (at least 2 inputs required). If filter = true, filter all input NaN to 0 before adding.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input
        filter (bool): If True, filter NaN to 0 before adding

    Returns:
        pd.Series: The sum of inputs
    """
    if filter:
        x = x.fillna(0)
        if isinstance(y, pd.Series):
            y = y.fillna(0)
    return x + y


def divide(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    x / y
    
    Args:
        x (pd.Series): The numerator series
        y (Union[pd.Series, float]): The denominator

    Returns:
        pd.Series: x / y
    """
    if isinstance(y, pd.Series):
        return x / (y + EPS)
    return x / (y + EPS)


def log(x: pd.Series) -> pd.Series:
    """
    Natural logarithm.
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The natural logarithm of the input series
    """
    return np.log(x)


def max(x: Union[pd.Series, float], y: Union[pd.Series, float], *args: Union[pd.Series, float]) -> pd.Series:
    """
    Element-wise maximum of 2 or more inputs.
    
    Args:
        x (Union[pd.Series, float]): The first input
        y (Union[pd.Series, float]): The second input
        *args: Union[pd.Series, float]: Additional inputs

    Returns:
        pd.Series: The element-wise maximum of the inputs
    """
    base = x if isinstance(x, pd.Series) else y
    idx = base.index

    arrs = [
        x.values if isinstance(x, pd.Series) else np.full(len(idx), x),
        y.values if isinstance(y, pd.Series) else np.full(len(idx), y),
    ]

    for a in args:
        arrs.append(
            a.values if isinstance(a, pd.Series) else np.full(len(idx), a)
        )

    return pd.Series(np.maximum.reduce(arrs), index=idx)


def min(x: Union[pd.Series, float], y: Union[pd.Series, float], *args: Union[pd.Series, float]) -> pd.Series:
    """
    Element-wise minimum of 2 or more inputs.
    
    Args:
        x (Union[pd.Series, float]): The first input
        y (Union[pd.Series, float]): The second input
        *args: Union[pd.Series, float]: Additional inputs

    Returns:
        pd.Series: The element-wise minimum of the inputs
    """
    base = x if isinstance(x, pd.Series) else y
    idx = base.index

    arrs = [
        x.values if isinstance(x, pd.Series) else np.full(len(idx), x),
        y.values if isinstance(y, pd.Series) else np.full(len(idx), y),
    ]

    for a in args:
        arrs.append(
            a.values if isinstance(a, pd.Series) else np.full(len(idx), a)
        )

    return pd.Series(np.minimum.reduce(arrs), index=idx)


def multiply(x: pd.Series, y: Union[pd.Series, float], filter: bool = False) -> pd.Series:
    """
    Multiply all inputs. At least 2 inputs are required. Filter sets the NaN values to 1.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input
        filter (bool): If True, set NaN values to 1

    Returns:
        pd.Series: The product of inputs
    """
    if filter:
        x = x.fillna(1)
        if isinstance(y, pd.Series):
            y = y.fillna(1)
    return x * y


def power(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    x ^ y
    
    Args:
        x (pd.Series): The base series
        y (Union[pd.Series, float]): The exponent

    Returns:
        pd.Series: x raised to the power of y
    """
    return np.power(x, y)


def reverse(x: pd.Series) -> pd.Series:
    """
    -x
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The negated series
    """
    return -x


def sign(x: pd.Series) -> pd.Series:
    """
    if input > 0, return 1; if input < 0, return -1; if input = 0, return 0; if input = NaN, return NaN.
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The sign of the input series
    """
    return np.sign(x)


def signed_power(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    x raised to the power of y such that final result preserves sign of x.
    
    Args:
        x (pd.Series): The base series
        y (Union[pd.Series, float]): The exponent

    Returns:
        pd.Series: sign(x) * abs(x)^y
    """
    return np.sign(x) * np.power(np.abs(x), y)


def sqrt(x: pd.Series) -> pd.Series:
    """
    Square root of x.
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The square root of the input series
    """
    return np.sqrt(x)


# ============ Logical Operators ============

def and_op(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    Logical AND operator, returns true if both operands are true and returns false otherwise.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input

    Returns:
        pd.Series: Logical AND result
    """
    return (x != 0) & (y != 0)


def if_else(x: pd.Series, y: pd.Series, z: Union[pd.Series, float]) -> pd.Series:
    """
    If x is true then return y else return z.
    
    Args:
        x (pd.Series): Boolean condition series
        y (pd.Series): Values to return when condition is True
        z (Union[pd.Series, float]): Values to return when condition is False

    Returns:
        pd.Series: The conditional result
    """
    return pd.Series(np.where(x != 0, y, z), index=x.index)


def lt(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    If x < y return true, else return false.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input

    Returns:
        pd.Series: Boolean series (True where x < y)
    """
    return x < y


def le(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    Returns true if x <= y, return false otherwise.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input

    Returns:
        pd.Series: Boolean series (True where x <= y)
    """
    return x <= y


def eq(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    Returns true if both inputs are same and returns false otherwise.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input

    Returns:
        pd.Series: Boolean series (True where x == y)
    """
    return x == y


def gt(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    Logic comparison operators to compares two inputs.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input

    Returns:
        pd.Series: Boolean series (True where x > y)
    """
    return x > y


def ge(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    Returns true if x >= y, return false otherwise.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input

    Returns:
        pd.Series: Boolean series (True where x >= y)
    """
    return x >= y


def ne(x: pd.Series, y: Union[pd.Series, float]) -> pd.Series:
    """
    Returns true if both inputs are NOT the same and returns false otherwise.
    
    Args:
        x (pd.Series): The first input series
        y (Union[pd.Series, float]): The second input

    Returns:
        pd.Series: Boolean series (True where x != y)
    """
    return x != y


def is_nan(x: pd.Series) -> pd.Series:
    """
    If (x == NaN) return 1 else return 0.
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: 1 where NaN, 0 otherwise
    """
    return x.isna().astype(float)


# ============ Time Series Operators ============

def delay(x: pd.Series, period: int) -> pd.Series:
    """
    Shift the series backward by the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): Number of periods to shift backward

    Returns:
        pd.Series: The delayed series
    """
    return x.shift(period)


def ts_delay(x: pd.Series, period: int) -> pd.Series:
    """
    Time series delay: shift the series backward by the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): Number of periods to shift backward

    Returns:
        pd.Series: The delayed series
    """
    return x.shift(period)


def delta(x: pd.Series, period: int = 1) -> pd.Series:
    """
    Difference between current value and value n periods ago.
    
    Args:
        x (pd.Series): The input series
        period (int): Number of periods to look back (default: 1)

    Returns:
        pd.Series: The difference series
    """
    return x - x.shift(period)


def ts_delta(x: pd.Series, period: int) -> pd.Series:
    """
    Time series delta: difference between current value and value n periods ago.
    
    Args:
        x (pd.Series): The input series
        period (int): Number of periods to look back

    Returns:
        pd.Series: The difference series
    """
    return x - x.shift(period)


def ts_rank(x: pd.Series, period: int) -> pd.Series:
    """
    Time series rank: rank of current value within the rolling window.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The time series rank (percentile rank)
    """
    return x.rolling(window=period).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)


def ts_sum(x: pd.Series, period: int) -> pd.Series:
    """
    Rolling sum over the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling sum
    """
    return x.rolling(window=period).sum()


def ts_mean(x: pd.Series, period: int) -> pd.Series:
    """
    Rolling mean over the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling mean
    """
    return x.rolling(window=period).mean()


def ts_max(x: pd.Series, period: int) -> pd.Series:
    """
    Rolling maximum over the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling maximum
    """
    return x.rolling(window=period).max()


def ts_min(x: pd.Series, period: int) -> pd.Series:
    """
    Rolling minimum over the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling minimum
    """
    return x.rolling(window=period).min()


def ts_stddev(x: pd.Series, period: int) -> pd.Series:
    """
    Rolling standard deviation over the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling standard deviation
    """
    return x.rolling(window=period).std()


def ts_std_dev(x: pd.Series, period: int) -> pd.Series:
    """
    Rolling standard deviation over the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling standard deviation
    """
    return x.rolling(window=period).std()


def ts_corr(x: pd.Series, y: pd.Series, period: int) -> pd.Series:
    """
    Returns correlation of x and y for the past d days.
    
    Args:
        x (pd.Series): The first input series
        y (pd.Series): The second input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling correlation
    """
    return x.rolling(window=period).corr(y)


def ts_covariance(x: pd.Series, y: pd.Series, period: int) -> pd.Series:
    """
    Rolling covariance between two series over the specified number of periods.
    
    Args:
        x (pd.Series): The first input series
        y (pd.Series): The second input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling covariance
    """
    return x.rolling(window=period).cov(y)


def ts_argmax(x: pd.Series, period: int) -> pd.Series:
    """
    Returns the relative index of the max value in the time series for the past d days.
    If the current day has the max value for the past d days, it returns 0.
    If previous day has the max value for the past d days, it returns 1.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The index of maximum value (0 to period-1)
    """
    def argmax_func(s):
        if len(s) == 0 or s.isna().all():
            return np.nan
        return len(s) - 1 - s.values.argmax()
    
    return x.rolling(window=period).apply(argmax_func, raw=True)


def ts_arg_min(x: pd.Series, period: int) -> pd.Series:
    """
    Returns the relative index of the min value in the time series for the past d days.
    If the current day has the min value for the past d days, it returns 0.
    If previous day has the min value for the past d days, it returns 1.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The index of minimum value (0 to period-1)
    """
    def argmin_func(s):
        if len(s) == 0 or s.isna().all():
            return np.nan
        return len(s) - 1 - s.values.argmin()
    
    return x.rolling(window=period).apply(argmin_func, raw=True)


def ts_product(x: pd.Series, period: int) -> pd.Series:
    """
    Rolling product over the specified number of periods.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling product
    """
    return x.rolling(window=period).apply(lambda s: s.prod(), raw=False)


def ts_zscore(x: pd.Series, period: int) -> pd.Series:
    """
    Time series z-score: (x - ts_mean(x, d)) / ts_std_dev(x, d).
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: The rolling z-score
    """
    mean_val = x.rolling(window=period).mean()
    std_val = x.rolling(window=period).std()
    return (x - mean_val) / (std_val + EPS)


def ts_av_diff(x: pd.Series, period: int) -> pd.Series:
    """
    Returns x - ts_mean(x, d), but deals with NaNs carefully.
    That is NaNs are ignored during mean computation.
    
    Args:
        x (pd.Series): The input series
        period (int): The rolling window size

    Returns:
        pd.Series: x - ts_mean(x, period)
    """
    mean_val = x.rolling(window=period).mean()
    return x - mean_val


def days_from_last_change(x: pd.Series) -> pd.Series:
    """
    Amount of days since last change of x.
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: Days since last change
    """
    result = pd.Series(index=x.index, dtype=float)
    for i in range(len(x)):
        if i == 0:
            result.iloc[i] = 0
        else:
            # Find the last index where value changed
            for j in range(i-1, -1, -1):
                if not pd.isna(x.iloc[j]) and not pd.isna(x.iloc[i]) and x.iloc[j] != x.iloc[i]:
                    result.iloc[i] = i - j
                    break
            else:
                result.iloc[i] = i  # No change found
    return result


def hump(x: pd.Series, hump: float = 0.01) -> pd.Series:
    """
    Limits amount and magnitude of changes in input (thus reducing turnover).
    
    Args:
        x (pd.Series): The input series
        hump (float): Hump parameter (default: 0.01)

    Returns:
        pd.Series: The humped series
    """
    # Simple implementation: clip changes to hump threshold
    diff = x.diff()
    clipped_diff = diff.clip(-hump, hump)
    return x.iloc[0] + clipped_diff.cumsum()


def kth_element(x: pd.Series, d: int, k: int) -> pd.Series:
    """
    Returns K-th valid value of input by looking through lookback days.
    This operator can be used to backfill missing data if k=1.
    
    Args:
        x (pd.Series): The input series
        d (int): Lookback days
        k (int): K-th element (1-indexed)

    Returns:
        pd.Series: K-th valid value
    """
    def kth_func(s):
        valid_values = s.dropna()
        if len(valid_values) >= k:
            return valid_values.iloc[k-1]
        return np.nan
    
    return x.rolling(window=d).apply(kth_func, raw=False)


def last_diff_value(x: pd.Series, d: int) -> pd.Series:
    """
    Returns last x value not equal to current x value from last d days.
    
    Args:
        x (pd.Series): The input series
        d (int): Lookback days

    Returns:
        pd.Series: Last different value
    """
    def last_diff_func(s):
        current = s.iloc[-1]
        for val in reversed(s.iloc[:-1]):
            if not pd.isna(val) and not pd.isna(current) and val != current:
                return val
        return np.nan
    
    return x.rolling(window=d).apply(last_diff_func, raw=False)


# ============ Vector Operators ============

def rank(x: pd.Series) -> pd.Series:
    """
    Cross-sectional rank of the input series (percentile rank).
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The rank (0 to 1) of each value
    """
    return x.rank(pct=True)


def zscore(x: pd.Series) -> pd.Series:
    """
    Z-score normalization: (x - mean(x)) / std(x).
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The z-score normalized series
    """
    mean_val = x.mean()
    std_val = x.std()
    if std_val == 0:
        return pd.Series(0, index=x.index)
    return (x - mean_val) / (std_val + EPS)


def quantile(x: pd.Series, percentage: float) -> pd.Series:
    """
    Quantile of the input series.
    
    Args:
        x (pd.Series): The input series
        percentage (float): Quantile value (0 to 1)

    Returns:
        pd.Series: The quantile value repeated for all elements
    """
    q_val = x.quantile(percentage)
    return pd.Series([q_val] * len(x), index=x.index)


def normalize(x: pd.Series) -> pd.Series:
    """
    Normalize the series to [0, 1] range.
    
    Args:
        x (pd.Series): The input series

    Returns:
        pd.Series: The normalized series
    """
    min_val = x.min()
    max_val = x.max()
    if max_val == min_val:
        return pd.Series(0.5, index=x.index)
    return (x - min_val) / (max_val - min_val + EPS)
