import warnings
from pandas import DataFrame

warnings.filterwarnings('ignore')

from typing import Any, Dict

import gym
import numpy as np
import pandas as pd

from src.registry import ENVIRONMENT
from src.utils import get_start_end_timestamp
from src.utils import TimeLevel
from src.logger import logger

__all__ = ['TradingEnvironment']


@ENVIRONMENT.register_module(force=True)
class TradingEnvironment(gym.Env):
    def __init__(
        self,
        *args,
        dataset: Any = None,
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        history_timestamps: int = 32,
        step_timestamps: int = 1,
        start_timestamp='2008-04-01',
        end_timestamp='2021-04-01',
        **kwargs,
    ):
        """
        Initialize the Trading environment.
        
        Args:
            dataset: SingleAssetDataset instance containing price and feature data
            initial_amount: Initial cash amount
            transaction_cost_pct: Transaction cost percentage (e.g., 0.001 for 0.1%)
            history_timestamps: Number of historical timestamps in each sample
            step_timestamps: Number of timestamps to step forward
            start_timestamp: Start timestamp for filtering data
            end_timestamp: End timestamp for filtering data
        """
        super(TradingEnvironment, self).__init__()

        self.dataset = dataset
        self.symbol = self.dataset.symbol
        self.symbol_info = self.dataset.symbol_info
        self.level = self.dataset.level
        self.level_format = self.dataset.level_format

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.start_timestamp, self.end_timestamp = get_start_end_timestamp(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            level=self.level
        )

        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps

        # Initialize data structures
        self._init_data()
        
        # Action space: SELL (-1), HOLD (0), BUY (+1)
        self.action_labels = ['SELL', 'HOLD', 'BUY']
        self.action_dim = len(self.action_labels)

    def _init_data(self):
        """
        Initialize data structures from the dataset.
        
        Uses the combined DataFrame (price + features) directly and builds
        timestamp information for the specified time range.
        """
        # Get DataFrame (price + features)
        self.df = self.dataset.data["df"]
        self.all_columns = self.dataset.metadata["columns"]["all_columns"]
        self.price_columns = self.dataset.metadata["columns"]["price_columns"]  # Still needed for price extraction
        
        # Build timestamp info from dataset metadata
        timestamp_info = {}
        for sample_id, sample_data in self.dataset.metadata['items'].items():
            window_info = sample_data["window_info"]
            start_ts = window_info["start_timestamp"]
            end_ts = window_info["end_timestamp"]
            
            # Filter by time range
            if end_ts >= self.start_timestamp and end_ts <= self.end_timestamp:
                timestamp_info[sample_id] = {
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                }
        
        if not timestamp_info:
            raise ValueError(f"No samples found in time range {self.start_timestamp} to {self.end_timestamp}")
        
        self.timestamp_info = timestamp_info
        self.timestamp_min_index = min(timestamp_info.keys())
        self.timestamp_max_index = max(timestamp_info.keys())
        self.timestamp_min = timestamp_info[self.timestamp_min_index]["start_timestamp"]
        self.timestamp_max = timestamp_info[self.timestamp_max_index]["end_timestamp"]
        
        self.num_timestamps = self.timestamp_max_index - self.timestamp_min_index + 1
        assert self.num_timestamps == len(timestamp_info), \
            f"num_timestamps {self.num_timestamps} != len(timestamp_info) {len(timestamp_info)}"

    def _get_dataitem(self,
                      df: DataFrame,
                      start_timestamp: Any,
                      end_timestamp: Any) -> DataFrame:
        """
        Extract DataFrame slice within the specified timestamp range.
        
        Args:
            df: DataFrame with timestamp index
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (inclusive)
            
        Returns:
            Filtered DataFrame containing only rows within the time range
        """
        # 使用 loc 切片代替 deepcopy，避免每次调用都深拷贝整个 DataFrame
        mask = (df.index >= start_timestamp) & (df.index <= end_timestamp)
        return df.loc[mask].copy()

    def _init_timestamp_index(self):
        return self.timestamp_min_index

    def get_timestamp_string(self, timestamp_index: int):
        end_timestamp = self.timestamp_info[timestamp_index]["end_timestamp"]
        end_timestamp_string = end_timestamp.strftime(self.level_format.value)
        return end_timestamp_string

    def get_value(self,
                  cash: float,
                  postition: float,
                  price: float):
        value = cash + postition * price
        return value

    def get_price(self, timestamp_index: int) -> float:
        """
        Get the closing price at the specified timestamp index.
        
        Args:
            timestamp_index: Index of the timestamp in timestamp_info
            
        Returns:
            Closing price (float)
        """
        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        
        df = self._get_dataitem(self.df, start_timestamp, end_timestamp)
        # Return the closing price of the last row
        return float(df.iloc[-1]["close"])

    def get_data(self) -> DataFrame:
        """
        Get all data within the specified time range.
        
        Returns:
            Combined DataFrame containing price and feature columns
        """
        start_timestamp_index = self.timestamp_min_index
        end_timestamp_index = self.timestamp_max_index

        # Calculate start timestamp based on time level
        if self.level == TimeLevel.DAY:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(days=1)
        elif self.level == TimeLevel.HOUR:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(hours=1)
        elif self.level == TimeLevel.MINUTE:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(minutes=1)
        else:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(seconds=1)

        end_timestamp = self.timestamp_info[end_timestamp_index]["end_timestamp"]

        # Return DataFrame
        return self._get_dataitem(self.df, start_timestamp, end_timestamp)

    def get_price_full(self, timestamp_index: int) -> tuple:
        """
        Get all price information (close, high, low, open, volume) at the specified timestamp index.
        
        Args:
            timestamp_index: Index of the timestamp in timestamp_info
            
        Returns:
            Tuple of (close, high, low, open, volume) prices
        """
        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        
        df = self._get_dataitem(self.df, start_timestamp, end_timestamp)
        prices = df[self.price_columns].iloc[-1].to_dict()
        
        return (
            float(prices["close"]),
            float(prices["high"]),
            float(prices["low"]),
            float(prices["open"]),
            float(prices["volume"])
        )

    def get_state(self, timestamp_index: int) -> Dict[str, Any]:
        """
        Get the current state at the specified timestamp index.
        
        Args:
            timestamp_index: Index of the timestamp in timestamp_info
            
        Returns:
            Dictionary containing:
                - timestamp: Formatted timestamp string
                - data: Combined DataFrame (price + features) for the current window
        """
        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        df = self._get_dataitem(self.df, start_timestamp, end_timestamp)

        return dict(
            timestamp=end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            data=df
        )

    def eval_buy_position(self,
                          cash: float,
                          price: float):
        # evaluate buy position (supports fractional positions for crypto)
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return cash / price / (1 + self.transaction_cost_pct)

    def eval_sell_position(self,
                           position: float):
        # evaluate sell position (supports fractional positions)
        return float(position)

    def buy(self,
            cash: float,
            position: float,
            price: float,
            amount: float):

        # evaluate buy position (maximum number of shares we can buy, supports fractional)
        eval_buy_postion = self.eval_buy_position(price=price, cash=cash)
        # logger.info(f"| 📊 buy(): cash={cash}, price={price}, eval_buy_position={eval_buy_postion}, amount={amount}")

        # predict buy position: amount is a ratio (0.0 to 1.0) of available buying power
        # Clamp amount to [0.0, 1.0] range
        amount = max(0.0, min(1.0, abs(amount)))
        buy_position = amount * eval_buy_postion  # Support fractional positions
        # logger.info(f"| 📊 buy(): buy_position={buy_position} (amount={amount} * eval_buy_position={eval_buy_postion})")

        cash = cash - (buy_position * price * (1 + self.transaction_cost_pct))
        position = position + buy_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if buy_position <= 1e-10:  # Use small epsilon for float comparison
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "BUY"
            action = self.action_labels.index("BUY")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def sell(self,
             cash: float,
             position: float,
             price: float,
             amount: float):

        # evaluate sell position (maximum number of shares we can sell, supports fractional)
        eval_sell_postion = self.eval_sell_position(position=position)

        # predict sell position: amount is a ratio (0.0 to 1.0) of current position
        # Clamp amount to [0.0, 1.0] range
        amount = max(0.0, min(1.0, abs(amount)))
        sell_position = amount * eval_sell_postion  # Support fractional positions

        cash = cash + (sell_position * price * (1 - self.transaction_cost_pct))
        position = position - sell_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if sell_position <= 1e-10:  # Use small epsilon for float comparison
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "SELL"
            action = self.action_labels.index("SELL")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def hold(self,
             cash: float,
             position: float,
             price: float,
             amount: float = 0.0):

        value = self.get_value(cash=cash, postition=position, price=price)

        action_label = "HOLD"
        action = self.action_labels.index("HOLD")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        self.ret = 0.0
        self.cash = self.initial_amount
        self.position = 0.0  # Support fractional positions for crypto
        self.discount = 1.0
        self.pre_value = self.value = self.initial_amount
        self.value = self.initial_amount
        self.total_return = 0.0
        self.total_profit = 0.0
        self.action = 1
        self.action_label = 'HOLD'
        self.done = False

        # after init record, get the state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            price=self.price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )

        return self.state, info

    def step(self, action: Any, position_ratio: float = 1.0):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take. Can be:
                - int: -1 (SELL), 0 (HOLD), 1 (BUY)
                - np.ndarray: Array containing action index
            position_ratio: Ratio of position to use (0.0 to 1.0), default 1.0 for full position
        
        Returns:
            Tuple of (state, reward, done, truncated, info)
        """
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, int):
            action = int(action)
        else:
            # If action is not recognized, default to HOLD
            action = 0
        
        # logger.info(f"| 📊 Environment.step: action={action}, position_ratio={position_ratio}, cash={self.cash}, price={self.price}, current_position={self.position}")

        if action > 0:
            res_info = self.buy(cash=self.cash,
                                position=self.position,
                                price=self.price,
                                amount=position_ratio)
        elif action < 0:
            res_info = self.sell(cash=self.cash,
                                 position=self.position,
                                 price=self.price,
                                 amount=abs(position_ratio))
        else:
            res_info = self.hold(cash=self.cash,
                                 position=self.position,
                                 price=self.price,
                                 amount=action)

        self.cash = res_info['cash']
        self.position = res_info['position']
        self.value = res_info['value']
        self.action = res_info['action']
        self.action_label = res_info['action_label']

        ret = (self.value - self.pre_value) / (self.pre_value + 1e-6)

        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        # next timestamp
        self.timestamp_index = self.timestamp_index + 1
        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncted = False
        else:
            self.done = True
            self.truncted = True

        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        # after update record, get the state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        reward = ret

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            price=self.price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )

        # update the pre_value
        self.pre_value = self.value

        return self.state, reward, self.done, self.truncted, info