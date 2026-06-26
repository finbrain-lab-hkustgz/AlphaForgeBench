from typing import Union
import numpy as np

def clean_invalid_values(arr: np.ndarray) -> np.ndarray:
    """Clean invalid values (NaN and infinite) from a NumPy array.
    
    Args:
        arr (np.ndarray): Input array.
    Returns:
        np.ndarray: Processed array with NaN and infinite values removed.
    """
    mask = np.isfinite(arr)
    arr = arr[mask]
    return arr

def fill_invalid_values(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Fill invalid values (NaN and infinite) in a NumPy array with a specified fill value.
    
    Args:
        arr (np.ndarray): Input array.
        fill_value (float): Value to fill invalid entries with. Default is 0.0.
    Returns:
        np.ndarray: Processed array with invalid values replaced by fill_value.
    """
    arr = np.nan_to_num(arr, nan=.0, posinf=1.0, neginf=.0)
    return arr