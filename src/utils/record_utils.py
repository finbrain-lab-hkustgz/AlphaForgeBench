import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class Record():
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    price: float
    cash: float
    position: int
    pre_value: Optional[float]
    action: Optional[str]
    post_value: Optional[float]
    ret: Optional[float]
    reason: Optional[str] = None

class TradingRecords():
    def __init__(self):
        self.data = dict(
            # state (action-before)
            timestamp = [],
            close = [],
            high = [],
            low = [],
            open = [],
            volume = [],
            price = [],
            position = [],
            cash = [],
            value = [],

            # action (action-after)
            action = [],
            action_label = [],
            ret = [],
            total_profit=[],
            reason = [],
        )

    def add(self, info: Dict[str, Any]):
        """
        Add a new record to the trading records.
        :param info: A dictionary containing the trading information.
        """
        for key, value in info.items():
            self.data[key].append(value)

    def to_dataframe(self):
        """
        Convert the trading records to a pandas DataFrame.
        :return: A pandas DataFrame containing the trading records.
        """
        # Check if we have any data
        if not self.data['timestamp']:
            return pd.DataFrame()
        
        # Filter out empty lists and ensure all lists have the same length
        filtered_data = {}
        base_length = len(self.data['timestamp'])
        
        for key, values in self.data.items():
            if len(values) == base_length:
                filtered_data[key] = values
            else: # fill the missing values with None
                filtered_data[key] = values + [None] * (base_length - len(values))
        
        df = pd.DataFrame(filtered_data, index=range(base_length))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df.set_index('timestamp', inplace=True)
        return df


class MultiTradingRecords():
    """Records for multi-asset trading strategies.

    Tracks trading activity across multiple symbols simultaneously,
    recording individual positions, actions, and overall portfolio value.
    """
    def __init__(self):
        self.data = dict(
            # state (action-before)
            timestamp = [],
            cash = [],
            value = [],

            # per-symbol data (Dict[symbol, value])
            prices = [],  # Dict[symbol, price]
            positions = [],  # Dict[symbol, position_count]
            actions = [],  # Dict[symbol, action_label]

            # portfolio-level metrics (action-after)
            ret = [],
            total_profit = [],
        )

    def add(self, info: Dict[str, Any]):
        """Add a new record to the multi-trading records.

        Args:
            info: Dictionary containing trading information with keys:
                - timestamp: str
                - cash: float
                - value: float
                - prices: Dict[symbol, price]
                - positions: Dict[symbol, position_count]
                - actions: Dict[symbol, action_label]
                - ret: float
                - total_profit: float
        """
        for key, value in info.items():
            if key in self.data:
                self.data[key].append(value)

    def to_dataframe(self):
        """Convert the multi-trading records to a pandas DataFrame.

        Returns:
            pandas DataFrame with timestamp index and columns for:
            - cash, value, ret, total_profit
            - per-symbol columns: {symbol}_price, {symbol}_position, {symbol}_action
        """
        # Check if we have any data
        if not self.data['timestamp']:
            return pd.DataFrame()

        # Build flattened data for DataFrame
        flattened_data = {
            'timestamp': self.data['timestamp'],
            'cash': self.data['cash'],
            'value': self.data['value'],
            'ret': self.data['ret'],
            'total_profit': self.data['total_profit'],
        }

        # Extract all unique symbols
        all_symbols = set()
        for prices_dict in self.data['prices']:
            if prices_dict:
                all_symbols.update(prices_dict.keys())

        # Create per-symbol columns
        for symbol in sorted(all_symbols):
            flattened_data[f'{symbol}_price'] = []
            flattened_data[f'{symbol}_position'] = []
            flattened_data[f'{symbol}_action'] = []

        # Fill per-symbol columns
        for i in range(len(self.data['timestamp'])):
            prices_dict = self.data['prices'][i] if i < len(self.data['prices']) else {}
            positions_dict = self.data['positions'][i] if i < len(self.data['positions']) else {}
            actions_dict = self.data['actions'][i] if i < len(self.data['actions']) else {}

            for symbol in sorted(all_symbols):
                flattened_data[f'{symbol}_price'].append(prices_dict.get(symbol, None))
                flattened_data[f'{symbol}_position'].append(positions_dict.get(symbol, 0))
                flattened_data[f'{symbol}_action'].append(actions_dict.get(symbol, 'HOLD'))

        # Create DataFrame
        df = pd.DataFrame(flattened_data, index=range(len(self.data['timestamp'])))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df.set_index('timestamp', inplace=True)
        return df


class PortfolioRecords():
    def __init__(self):
        self.data = dict(
            # state (action-before)
            timestamp = [],
            price = [],
            position = [],
            cash = [],
            value = [],

            # action (action-after)
            action = [],
            ret = [],
            total_profit = [],
            reason = [],
        )

    def add(self, info: Dict[str, Any]):
        """
        Add a new record to the portfolio records.
        :param info: A dictionary containing the portfolio information.
        """
        for key, value in info.items():
            self.data[key].append(value)

    def to_dataframe(self):
        """
        Convert the portfolio records to a pandas DataFrame.
        :return: A pandas DataFrame containing the portfolio records.
        """
        df = pd.DataFrame(self.data, index=range(len(self.data['timestamp'])))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df.set_index('timestamp', inplace=True)
        return df