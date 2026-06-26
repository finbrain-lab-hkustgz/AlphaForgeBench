import warnings
from typing import Any, Dict, Optional, List

import gym
from pandas import DataFrame

from src.registry import ENVIRONMENT
from src.utils import get_start_end_timestamp

warnings.filterwarnings('ignore')

__all__ = ['FactorEnvironment']


@ENVIRONMENT.register_module(force=True)
class FactorEnvironment(gym.Env):
    """
    A simple environment for factor analysis.
    
    This environment only requires OHLCV (Open, High, Low, Close, Volume) price data.
    It provides a clean interface for factor computation without the complexity
    of trading simulation.
    """
    
    def __init__(
        self,
        *args,
        dataset: Any = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
        return_periods: List[int] = [1, 5, 10],
        **kwargs,
    ):
        """
        Initialize the FactorEnvironment.
        
        Args:
            dataset: SingleAssetDataset instance containing price data
            start_timestamp: Start timestamp for filtering data (optional)
            end_timestamp: End timestamp for filtering data (optional)
            return_periods: List of periods for calculating forward returns (e.g., [1, 5, 10] 
                           will calculate ret_1, ret_5, ret_10)
        """
        super(FactorEnvironment, self).__init__()
        
        self.dataset = dataset
        self.symbol = self.dataset.symbol
        self.symbol_info = self.dataset.symbol_info
        self.level = self.dataset.level
        self.level_format = self.dataset.level_format
        
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        
        # Return periods
        self.return_periods = return_periods
        
        # Initialize OHLCV data
        self._init_data()
    
    def _init_data(self):
        """
        Initialize OHLCV DataFrame from the dataset and calculate forward returns.
        
        This method:
        1. Extracts OHLCV columns from the dataset
        2. Filters data by time range if specified
        3. Calculates forward returns for each period in return_periods
           (e.g., ret_1, ret_5, ret_10 for periods [1, 5, 10])
        """
        # Get DataFrame from dataset
        full_df = self.dataset.data["df"]
        price_columns = self.dataset.data["price_columns"]
        
        # Extract only OHLCV columns
        self.df = full_df[price_columns].copy()
        
        # Filter by time range if specified
        if self.start_timestamp or self.end_timestamp:
            start_ts, end_ts = get_start_end_timestamp(
                start_timestamp=self.start_timestamp,
                end_timestamp=self.end_timestamp,
                level=self.level
            )
            
            if start_ts:
                self.df = self.df[self.df.index >= start_ts]
            if end_ts:
                self.df = self.df[self.df.index <= end_ts]
        
        # Ensure data is sorted by timestamp
        self.df = self.df.sort_index()
        
        # Validate OHLCV columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in expected_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required OHLCV columns: {missing_columns}")
        
        # Calculate returns for each return period
        close = self.df['close']
        for period in self.return_periods:
            if period > 0:
                # Calculate forward return: (close_future - close_current) / close_current
                future_close = close.shift(-period)
                ret = (future_close - close) / close
                self.df[f'ret_{period}'] = ret
        
        # Store column names for reference
        self.price_columns = price_columns
        self.return_columns = [f'ret_{period}' for period in self.return_periods]
    
    def reset(self, **kwargs) -> Dict[str, Any]:
        """
        Reset the environment.
        
        Returns:
            Dictionary containing:
                - df: OHLCV DataFrame
                - symbol: Symbol identifier
                - info: Additional information
        """
        info = dict(
            symbol=self.symbol,
            shape=self.df.shape,
            columns=list(self.df.columns),
            start_timestamp=self.df.index[0] if len(self.df) > 0 else None,
            end_timestamp=self.df.index[-1] if len(self.df) > 0 else None,
            num_rows=len(self.df),
            return_periods=self.return_periods,
            return_columns=self.return_columns,
        )
        
        state = self.get_data()
        
        return state, info
    
    def get_data(self) -> DataFrame:
        """
        Get the OHLCV DataFrame.
        
        Returns:
            DataFrame containing OHLCV data
        """
        return self.df.copy()
    
    def get_price_data(self, start_timestamp: Any = None, end_timestamp: Any = None) -> DataFrame:
        """
        Get OHLCV data within a specific time range.
        
        Args:
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (inclusive)
            
        Returns:
            Filtered DataFrame containing OHLCV data
        """
        df = self.df.copy()
        
        if start_timestamp:
            df = df[df.index >= start_timestamp]
        if end_timestamp:
            df = df[df.index <= end_timestamp]
        
        return df
    
    def __len__(self):
        """Return the number of rows in the DataFrame."""
        return len(self.df)
    
    def __str__(self):
        """Return string representation of the environment."""
        info_str = f"{'-' * 50} FactorEnvironment {'-' * 50}\n"
        info_str += f"- Symbol: {self.symbol}\n"
        info_str += f"- Shape: {self.df.shape}\n"
        info_str += f"- Columns: {list(self.df.columns)}\n"
        if len(self.df) > 0:
            info_str += f"- Start: {self.df.index[0]}\n"
            info_str += f"- End: {self.df.index[-1]}\n"
        info_str += f"{'-' * 50} FactorEnvironment {'-' * 50}\n"
        return info_str

    def step(self, action: Any):
        """
        Take a step in the environment.
        """
        
        state = self.get_data()
        
        reward = 0.0
        done = False
        truncated = False
        info = dict(
            symbol=self.symbol,
            shape=self.df.shape,
            columns=list(self.df.columns),
            start_timestamp=self.df.index[0] if len(self.df) > 0 else None,
            end_timestamp=self.df.index[-1] if len(self.df) > 0 else None,
            num_rows=len(self.df),
            return_periods=self.return_periods,
            return_columns=self.return_columns,
        )
        
        return state, reward, done, truncated, info