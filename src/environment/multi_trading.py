"""
Multi-Trading Environment Module

This module provides an environment for trading multiple assets simultaneously
with a shared capital pool. Follows the same architecture as TradingEnvironment.
"""

import warnings
from copy import deepcopy
from typing import Any, Dict

import gym
import numpy as np
import pandas as pd
from pandas import DataFrame

warnings.filterwarnings('ignore')

from src.registry import ENVIRONMENT
from src.utils import get_start_end_timestamp, TimeLevel
from src.logger import logger

__all__ = ['MultiTradingEnvironment']


@ENVIRONMENT.register_module(force=True)
class MultiTradingEnvironment(gym.Env):
    """Multi-asset trading environment with shared capital pool.

    Follows the same architecture as TradingEnvironment but supports
    simultaneous trading of multiple symbols with independent signals
    and positions for each symbol.

    Key Features:
    - Shared cash pool across all symbols
    - Independent buy/sell signals per symbol
    - Independent position ratios per symbol
    - Support for empty positions (all cash)
    - Support for partial cash idle (position sum < 1.0)
    """

    def __init__(
        self,
        *args,
        dataset: Any = None,
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        max_position_pct: float = 0.5,
        history_timestamps: int = 32,
        step_timestamps: int = 1,
        start_timestamp='2008-04-01',
        end_timestamp='2021-04-01',
        **kwargs,
    ):
        """Initialize the Multi-Trading environment.

        Args:
            dataset: MultiAssetDataset instance containing price and feature data
            initial_amount: Initial cash amount
            transaction_cost_pct: Transaction cost percentage (e.g., 0.001 for 0.1%)
            max_position_pct: Maximum position percentage for a single symbol (e.g., 0.5 for 50%)
            history_timestamps: Number of historical timestamps in each sample
            step_timestamps: Number of timestamps to step forward
            start_timestamp: Start timestamp for filtering data
            end_timestamp: End timestamp for filtering data
        """
        super(MultiTradingEnvironment, self).__init__()

        self.dataset = dataset
        self.symbols = self.dataset.symbols  # List of symbols
        self.assets_info = self.dataset.assets_info
        self.level = self.dataset.level
        self.level_format = self.dataset.level_format

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_pct = max_position_pct

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

        # Action space: Each symbol can have SELL (-1), HOLD (0), BUY (+1)
        self.action_labels = ['SELL', 'HOLD', 'BUY']
        self.action_dim = len(self.action_labels)

    def _init_data(self):
        """Initialize data structures from the dataset.

        Loads multi-asset data and builds timestamp information for the
        specified time range.
        """
        # Get data from dataset - assets_data is Dict[symbol, Dict[data_type, DataFrame]]
        self.assets_data = self.dataset.assets_data

        # Get metadata for timestamp info - use per-symbol metadata
        self.assets_meta_info_items = self.dataset.assets_meta_info_items

        # Build timestamp info from dataset metadata
        # Use the assets_meta_info which has global timestamp information
        assets_meta = self.dataset.assets_meta_info

        timestamp_info = {}
        for sample_id, sample_data in assets_meta['items'].items():
            window_info = sample_data["history_info"]
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
        """Extract DataFrame slice within the specified timestamp range.

        Args:
            df: DataFrame with timestamp index
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (inclusive)

        Returns:
            Filtered DataFrame containing only rows within the time range
        """
        df_copy = deepcopy(df)
        return df_copy[(df_copy.index >= start_timestamp) & (df_copy.index <= end_timestamp)]

    def _init_timestamp_index(self):
        """Get initial timestamp index."""
        return self.timestamp_min_index

    def get_timestamp_string(self, timestamp_index: int):
        """Get formatted timestamp string."""
        end_timestamp = self.timestamp_info[timestamp_index]["end_timestamp"]
        end_timestamp_string = end_timestamp.strftime(self.level_format.value)
        return end_timestamp_string

    def get_value(self,
                  cash: float,
                  positions: Dict[str, float],
                  prices: Dict[str, float]):
        """Calculate total portfolio value.

        Args:
            cash: Available cash
            positions: Dict mapping symbol to position count
            prices: Dict mapping symbol to current price

        Returns:
            Total portfolio value
        """
        position_value = sum(positions.get(symbol, 0) * prices.get(symbol, 0)
                            for symbol in self.symbols)
        value = cash + position_value
        return value

    def get_prices(self, timestamp_index: int) -> Dict[str, float]:
        """Get closing prices for all symbols at the specified timestamp.

        Args:
            timestamp_index: Index of the timestamp in timestamp_info

        Returns:
            Dict mapping symbol to closing price
        """
        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]

        prices = {}
        for symbol in self.symbols:
            df = self.assets_data[symbol]["prices"]
            df_slice = self._get_dataitem(df, start_timestamp, end_timestamp)
            prices[symbol] = float(df_slice.iloc[-1]["close"])

        return prices

    def get_state(self, timestamp_index: int) -> Dict[str, Any]:
        """Get the current state at the specified timestamp index.

        Args:
            timestamp_index: Index of the timestamp in timestamp_info

        Returns:
            Dictionary containing:
                - timestamp: Formatted timestamp string
                - data: Dict mapping symbol to DataFrame (price + features)
        """
        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        # Build data dict for all symbols
        data = {}
        for symbol in self.symbols:
            # Get combined price + features DataFrame for this symbol
            df_prices = self.assets_data[symbol]["prices"]
            df_slice = self._get_dataitem(df_prices, start_timestamp, end_timestamp)

            # If features exist, merge them
            if "features" in self.assets_data[symbol]:
                df_features = self.assets_data[symbol]["features"]
                df_features_slice = self._get_dataitem(df_features, start_timestamp, end_timestamp)
                # Drop duplicate columns from features (features already contains price columns)
                feature_only_cols = [col for col in df_features_slice.columns if col not in df_slice.columns]
                df_combined = pd.concat([df_slice, df_features_slice[feature_only_cols]], axis=1)
            else:
                df_combined = df_slice

            data[symbol] = df_combined

        return dict(
            timestamp=end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            data=data
        )

    def _validate_action(self, action: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Validate and constrain action dictionary.

        Three-step validation:
        1. Filter invalid symbols
        2. Clamp signals and positions
        3. Enforce capital constraints

        Args:
            action: Dict mapping symbol to {"signal": int, "position": float}

        Returns:
            Validated action dictionary
        """
        if not isinstance(action, dict):
            logger.warning(f"| ⚠️ Action must be dict, got {type(action)}, defaulting to HOLD all")
            return {symbol: {"signal": 0, "position": 0.0} for symbol in self.symbols}

        validated_action = {}

        # Step 1: Filter invalid symbols
        for symbol in self.symbols:
            if symbol not in action:
                # Default to HOLD if symbol not in action
                validated_action[symbol] = {"signal": 0, "position": 0.0}
            else:
                symbol_action = action[symbol]
                if not isinstance(symbol_action, dict):
                    logger.warning(f"| ⚠️ Action for {symbol} must be dict, got {type(symbol_action)}, defaulting to HOLD")
                    validated_action[symbol] = {"signal": 0, "position": 0.0}
                    continue

                # Step 2: Clamp signal to [-1, 0, 1] and position to [0.0, max_position_pct]
                signal = symbol_action.get("signal", 0)
                position = symbol_action.get("position", 0.0)

                # Clamp signal
                if signal > 0:
                    signal = 1
                elif signal < 0:
                    signal = -1
                else:
                    signal = 0

                # Clamp position to [0.0, max_position_pct]
                position = max(0.0, min(self.max_position_pct, abs(position)))

                validated_action[symbol] = {"signal": signal, "position": position}

        # Step 3: Enforce capital constraints
        validated_action = self._enforce_capital_constraints(validated_action)

        return validated_action

    def _enforce_capital_constraints(self, action: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Scale down buy positions if total exceeds available cash.

        Important: Position sum can be < 1.0
        - If total buy demand = 60% cash, remaining 40% stays as cash
        - Does NOT force full position, allows strategy to keep cash reserve
        - Only scales down when total buy > available cash

        Args:
            action: Validated action dictionary

        Returns:
            Action dictionary with capital constraints enforced
        """
        # Calculate total buy demand
        total_buy_demand = 0.0
        for symbol, symbol_action in action.items():
            if symbol_action["signal"] == 1:  # BUY signal
                total_buy_demand += symbol_action["position"]

        # If total buy demand exceeds 100% of cash, scale down proportionally
        if total_buy_demand > 1.0:
            scale_factor = 1.0 / total_buy_demand
            logger.warning(f"| ⚠️ Total buy demand {total_buy_demand:.2%} exceeds available cash, scaling down by {scale_factor:.2f}")

            for symbol, symbol_action in action.items():
                if symbol_action["signal"] == 1:
                    action[symbol]["position"] = symbol_action["position"] * scale_factor

        return action

    def _execute_trade_for_symbol(self,
                                   symbol: str,
                                   signal: int,
                                   position_ratio: float,
                                   price: float) -> Dict[str, Any]:
        """Execute trade for a single symbol.

        Args:
            symbol: Symbol to trade
            signal: -1 (SELL), 0 (HOLD), 1 (BUY)
            position_ratio: Position ratio (0.0 to max_position_pct)
            price: Current price

        Returns:
            Dict with action_label and position_change
        """
        if signal > 0:  # BUY
            # Calculate how much we can buy with position_ratio of available cash
            buy_amount = position_ratio * self.cash
            buy_position = buy_amount / price / (1 + self.transaction_cost_pct)

            if buy_position <= 1e-10:
                return {"action_label": "HOLD", "position_change": 0.0, "cash_change": 0.0}

            # Update cash and position
            cost = buy_position * price * (1 + self.transaction_cost_pct)
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) + buy_position

            return {"action_label": "BUY", "position_change": buy_position, "cash_change": -cost}

        elif signal < 0:  # SELL
            current_position = self.positions.get(symbol, 0.0)
            if current_position <= 1e-10:
                return {"action_label": "HOLD", "position_change": 0.0, "cash_change": 0.0}

            # Calculate how much to sell (position_ratio of current position)
            sell_position = position_ratio * current_position

            if sell_position <= 1e-10:
                return {"action_label": "HOLD", "position_change": 0.0, "cash_change": 0.0}

            # Update cash and position
            proceeds = sell_position * price * (1 - self.transaction_cost_pct)
            self.cash += proceeds
            self.positions[symbol] = current_position - sell_position

            return {"action_label": "SELL", "position_change": -sell_position, "cash_change": proceeds}

        else:  # HOLD
            return {"action_label": "HOLD", "position_change": 0.0, "cash_change": 0.0}

    def reset(self, **kwargs):
        """Reset environment to initial state.

        Returns:
            Tuple of (state, info) following gym.Env interface
        """
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.prices = self.get_prices(timestamp_index=self.timestamp_index)

        self.ret = 0.0
        self.cash = self.initial_amount
        self.positions = {symbol: 0.0 for symbol in self.symbols}  # Dict[symbol, position_count]
        self.discount = 1.0
        self.pre_value = self.value = self.initial_amount
        self.value = self.initial_amount
        self.total_return = 0.0
        self.total_profit = 0.0
        self.actions = {symbol: "HOLD" for symbol in self.symbols}  # Dict[symbol, action_label]
        self.done = False

        # Get initial state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            prices=self.prices.copy(),
            cash=self.cash,
            positions=self.positions.copy(),
            actions=self.actions.copy(),
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            done=self.done,
        )

        return self.state, info

    def step(self, action: Dict[str, Dict[str, float]]):
        """Execute one step in the environment.

        Args:
            action: Dict mapping symbol to {"signal": int, "position": float}
                   signal: -1 (SELL), 0 (HOLD), 1 (BUY)
                   position: ratio of available capital/position to use

        Returns:
            Tuple of (state, reward, done, truncated, info)
        """
        # Validate action
        validated_action = self._validate_action(action)

        logger.info(f"| 📊 MultiTradingEnvironment.step: action={validated_action}, cash={self.cash}, prices={self.prices}")

        # Execute trades for each symbol
        actions_executed = {}
        for symbol in self.symbols:
            symbol_action = validated_action[symbol]
            signal = symbol_action["signal"]
            position_ratio = symbol_action["position"]
            price = self.prices[symbol]

            result = self._execute_trade_for_symbol(symbol, signal, position_ratio, price)
            actions_executed[symbol] = result["action_label"]

        self.actions = actions_executed

        # Calculate new portfolio value
        self.value = self.get_value(self.cash, self.positions, self.prices)

        # Calculate return
        ret = (self.value - self.pre_value) / (self.pre_value + 1e-6)
        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        # Move to next timestamp
        self.timestamp_index = self.timestamp_index + 1
        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncated = False
        else:
            self.done = True
            self.truncated = True

        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.prices = self.get_prices(timestamp_index=self.timestamp_index)

        # Get new state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        reward = ret

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            prices=self.prices.copy(),
            cash=self.cash,
            positions=self.positions.copy(),
            actions=self.actions.copy(),
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            done=self.done,
        )

        # Update pre_value for next step
        self.pre_value = self.value

        return self.state, reward, self.done, self.truncated, info
