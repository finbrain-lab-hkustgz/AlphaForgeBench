"""
Portfolio Environment Module

This module provides an environment for portfolio management with weight-based
asset allocation and periodic rebalancing. Unlike MultiTradingEnvironment which
uses independent signals per asset, PortfolioEnvironment manages the portfolio
as a whole with target weight allocations.

Key Differences from MultiTradingEnvironment:
- Action: Target weights (Dict[symbol, float]) instead of signals
- Execution: Rebalancing to target weights instead of incremental trades
- State: Includes portfolio-level information (current weights, correlations)
- Use Case: Factor-driven portfolio optimization, periodic rebalancing
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

import gym
import numpy as np
import pandas as pd
from pandas import DataFrame

warnings.filterwarnings('ignore')

from src.registry import ENVIRONMENT
from src.utils import get_start_end_timestamp, TimeLevel
from src.logger import logger

__all__ = ['PortfolioEnvironment']


@ENVIRONMENT.register_module(force=True)
class PortfolioEnvironment(gym.Env):
    """Portfolio management environment with weight-based rebalancing.

    This environment manages a portfolio of multiple assets with target weight
    allocation. It supports periodic rebalancing where the entire portfolio is
    adjusted to match target weights.

    Key Features:
    - Weight-based action space: {symbol: target_weight}
    - Automatic rebalancing to target weights
    - Portfolio-level state information (weights, correlations)
    - Support for cash allocation (weights sum can be < 1.0)
    - Factor-driven weight calculation compatible

    Action Format:
        {
            "BTCUSDT": 0.4,   # Target 40% of portfolio value
            "ETHUSDT": 0.3,   # Target 30% of portfolio value
            # Remaining 30% stays as cash
        }

    State Format:
        {
            "assets_data": {symbol: DataFrame},       # Price and feature data
            "current_weights": {symbol: weight},      # Current portfolio weights
            "current_prices": {symbol: price},        # Current prices
            "portfolio_value": float,                 # Total portfolio value
        }
    """

    def __init__(
        self,
        *args,
        dataset: Any = None,
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        max_position_pct: float = 1.0,  # Allow up to 100% in single asset for portfolio
        history_timestamps: int = 32,
        step_timestamps: int = 1,
        start_timestamp: str = '2008-04-01',
        end_timestamp: str = '2021-04-01',
        rebalance_frequency: int = 1,  # Rebalance every N steps (1 = every step)
        **kwargs,
    ):
        """Initialize the Portfolio environment.

        Args:
            dataset: MultiAssetDataset instance containing price and feature data
            initial_amount: Initial cash amount
            transaction_cost_pct: Transaction cost percentage (e.g., 0.001 for 0.1%)
            max_position_pct: Maximum position percentage for a single symbol
            history_timestamps: Number of historical timestamps in each sample
            step_timestamps: Number of timestamps to step forward
            start_timestamp: Start timestamp for filtering data
            end_timestamp: End timestamp for filtering data
            rebalance_frequency: Number of steps between rebalances (1 = every step)
        """
        super(PortfolioEnvironment, self).__init__()

        self.dataset = dataset
        self.symbols = self.dataset.symbols  # List of symbols
        self.assets_info = self.dataset.assets_info
        self.level = self.dataset.level
        self.level_format = self.dataset.level_format

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_pct = max_position_pct
        self.rebalance_frequency = rebalance_frequency

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

    def _init_data(self):
        """Initialize data structures from the dataset.

        Loads multi-asset data and builds timestamp information for the
        entire time series.
        """
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
        self.timestamp_list = sorted(list(self.timestamp_info.keys()))

        # Find valid timestamp range
        self.timestamp_min_index = self.timestamp_list[0]
        self.timestamp_max_index = self.timestamp_list[-1]
        self.num_timestamps = len(self.timestamp_list)

        # Load multi-asset data
        self.assets_data = {}
        for symbol in self.symbols:
            self.assets_data[symbol] = {
                'prices': self.dataset.assets_data[symbol]['prices'],
                'features': self.dataset.assets_data[symbol].get('features', None),
            }

        logger.info(
            f"| Portfolio Environment initialized: {len(self.symbols)} symbols, "
            f"{self.num_timestamps} timestamps"
        )

    def reset(self, **kwargs) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state.

        Returns:
            state: Dictionary containing portfolio state
            info: Dictionary containing metadata
        """
        # Reset to start of time series
        self.current_timestamp_idx = 0
        self.current_step = 0

        # Initialize portfolio state
        self.cash = self.initial_amount
        self.positions = {symbol: 0.0 for symbol in self.symbols}  # Number of shares
        self.portfolio_value = self.initial_amount

        # Get initial state using timestamp index
        state = self.get_state(self.current_timestamp_idx)

        # Build info
        current_timestamp_key = self.timestamp_list[self.current_timestamp_idx]
        info = {
            'timestamp': current_timestamp_key,
            'timestamp_index': self.current_timestamp_idx,
            'step': self.current_step,
            'symbols': self.symbols,
            'initial_value': self.initial_amount,
        }

        return state, info

    def step(self, action: Dict[str, float]) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step with target weight rebalancing.

        Args:
            action: Dictionary mapping symbol to target weight
                   Example: {"BTCUSDT": 0.4, "ETHUSDT": 0.3}
                   Weights should sum to <= 1.0 (remainder is cash)

        Returns:
            state: Updated portfolio state
            reward: Step reward (portfolio return)
            done: Whether episode is finished
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Validate action
        action = self._validate_action(action)

        # Check if it's time to rebalance
        should_rebalance = (self.current_step % self.rebalance_frequency == 0)

        # Get current prices using timestamp index
        current_prices = self.get_prices(self.current_timestamp_idx)

        # Calculate portfolio value before rebalancing
        value_before = self.get_value(self.cash, self.positions, current_prices)

        # Execute rebalancing if needed
        if should_rebalance:
            self._execute_rebalancing(action, current_prices)

        # Move to next timestamp
        self.current_timestamp_idx += self.step_timestamps
        self.current_step += 1

        # Check if episode is done
        done = self.current_timestamp_idx >= len(self.timestamp_list)
        truncated = False

        if not done:
            # Get new prices after step using new timestamp index
            next_prices = self.get_prices(self.current_timestamp_idx)

            # Calculate portfolio value after step
            value_after = self.get_value(self.cash, self.positions, next_prices)
            self.portfolio_value = value_after

            # Calculate reward (portfolio return)
            reward = (value_after - value_before) / value_before if value_before > 0 else 0.0

            # Get new state using new timestamp index
            state = self.get_state(self.current_timestamp_idx)

            # Get timestamp key for info
            next_timestamp_key = self.timestamp_list[self.current_timestamp_idx]
        else:
            reward = 0.0
            # Episode ended, return empty state
            state = {
                'assets_data': {},
                'current_weights': {},
                'current_prices': {},
                'portfolio_value': self.portfolio_value,
            }
            next_timestamp_key = self.timestamp_list[self.current_timestamp_idx - 1]

        # Build info
        info = {
            'timestamp': next_timestamp_key,
            'timestamp_index': self.current_timestamp_idx,
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'rebalanced': should_rebalance,
            'action': action,
            'reward': reward,
        }

        return state, reward, done, truncated, info

    def _validate_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize action weights.

        Args:
            action: Dictionary of target weights

        Returns:
            Validated action dictionary
        """
        # Ensure all symbols are present (default to 0)
        validated_action = {symbol: 0.0 for symbol in self.symbols}
        validated_action.update(action)

        # Check weight sum
        total_weight = sum(validated_action.values())
        if total_weight > 1.0:
            # Normalize to sum to 1.0
            logger.warning(
                f"| Action weights sum to {total_weight:.4f} > 1.0, normalizing"
            )
            validated_action = {
                symbol: weight / total_weight
                for symbol, weight in validated_action.items()
            }

        # Enforce max position constraint
        for symbol, weight in validated_action.items():
            if weight > self.max_position_pct:
                logger.warning(
                    f"| {symbol} weight {weight:.4f} exceeds max {self.max_position_pct:.4f}, capping"
                )
                validated_action[symbol] = self.max_position_pct

        return validated_action

    def _execute_rebalancing(self, target_weights: Dict[str, float], prices: Dict[str, float]):
        """Execute portfolio rebalancing to target weights.

        Args:
            target_weights: Dictionary of target weights per symbol
            prices: Current prices per symbol
        """
        # Calculate current portfolio value
        current_value = self.get_value(self.cash, self.positions, prices)

        # Calculate target value for each asset
        target_values = {
            symbol: target_weights[symbol] * current_value
            for symbol in self.symbols
        }

        # Calculate current value for each asset
        current_values = {
            symbol: self.positions[symbol] * prices[symbol]
            for symbol in self.symbols
        }

        # Calculate required trades
        trades = {}
        total_cost = 0.0

        for symbol in self.symbols:
            target_val = target_values[symbol]
            current_val = current_values[symbol]
            price = prices[symbol]

            # Calculate shares to trade
            delta_value = target_val - current_val
            delta_shares = delta_value / price if price > 0 else 0.0

            if abs(delta_shares) > 1e-8:  # Avoid tiny trades
                trades[symbol] = delta_shares
                # Calculate transaction cost
                trade_value = abs(delta_shares * price)
                total_cost += trade_value * self.transaction_cost_pct

        # Execute trades
        for symbol, delta_shares in trades.items():
            price = prices[symbol]
            self.positions[symbol] += delta_shares

            # Deduct cost from cash (both trade value and transaction cost)
            trade_value = delta_shares * price
            trade_cost = abs(trade_value) * self.transaction_cost_pct
            self.cash -= (trade_value + trade_cost)

        logger.debug(
            f"| Rebalanced portfolio: {len(trades)} trades, "
            f"total cost: ${total_cost:.2f}"
        )

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
        """Get current portfolio state at the specified timestamp index.

        Args:
            timestamp_index: Index of the timestamp in timestamp_info

        Returns:
            Dictionary containing:
            - assets_data: Historical price and feature data per symbol
            - current_weights: Current portfolio weights
            - current_prices: Current prices
            - portfolio_value: Total portfolio value
        """
        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        # Build data dict for all symbols
        assets_data = {}
        for symbol in self.symbols:
            # Get combined price + features DataFrame for this symbol
            df_prices = self.assets_data[symbol]["prices"]
            df_slice = self._get_dataitem(df_prices, start_timestamp, end_timestamp)

            # If features exist, merge them
            if "features" in self.assets_data[symbol] and self.assets_data[symbol]["features"] is not None:
                df_features = self.assets_data[symbol]["features"]
                df_features_slice = self._get_dataitem(df_features, start_timestamp, end_timestamp)
                # Drop duplicate columns from features (features already contains price columns)
                feature_only_cols = [col for col in df_features_slice.columns if col not in df_slice.columns]
                df_combined = pd.concat([df_slice, df_features_slice[feature_only_cols]], axis=1)
            else:
                df_combined = df_slice

            assets_data[symbol] = df_combined

        # Get current prices
        current_prices = self.get_prices(timestamp_index)

        # Calculate current weights
        current_value = self.get_value(self.cash, self.positions, current_prices)
        current_weights = {}
        if current_value > 0:
            for symbol in self.symbols:
                position_value = self.positions[symbol] * current_prices[symbol]
                current_weights[symbol] = position_value / current_value
        else:
            current_weights = {symbol: 0.0 for symbol in self.symbols}

        state = {
            'assets_data': assets_data,
            'current_weights': current_weights,
            'current_prices': current_prices,
            'portfolio_value': current_value,
        }

        return state

    def _get_dataitem(self, df: DataFrame, start_timestamp: Any, end_timestamp: Any) -> DataFrame:
        """Extract data slice between timestamps.

        Args:
            df: DataFrame to slice
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (inclusive)

        Returns:
            Sliced DataFrame
        """
        df_copy = df.copy()
        df_slice = df_copy[(start_timestamp <= df_copy.index) & (df_copy.index <= end_timestamp)]
        return df_slice

    def get_value(self, cash: float, positions: Dict[str, float], prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.

        Args:
            cash: Cash amount
            positions: Dictionary of positions (number of shares)
            prices: Dictionary of current prices

        Returns:
            Total portfolio value
        """
        position_value = sum(
            positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.symbols
        )
        value = cash + position_value
        return value

    def get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights.

        Returns:
            Dictionary mapping symbol to weight
        """
        current_prices = self.get_prices(self.current_timestamp_idx)
        current_value = self.get_value(self.cash, self.positions, current_prices)

        weights = {}
        if current_value > 0:
            for symbol in self.symbols:
                position_value = self.positions[symbol] * current_prices[symbol]
                weights[symbol] = position_value / current_value
        else:
            weights = {symbol: 0.0 for symbol in self.symbols}

        return weights

    def __len__(self):
        """Return the number of timestamps in the environment."""
        return self.num_timestamps

    def __str__(self):
        """Return string representation of the environment."""
        info_str = f"{'-' * 50} PortfolioEnvironment {'-' * 50}\n"
        info_str += f"- Symbols: {self.symbols}\n"
        info_str += f"- Timestamps: {self.num_timestamps}\n"
        info_str += f"- Initial Amount: ${self.initial_amount:.2f}\n"
        info_str += f"- Transaction Cost: {self.transaction_cost_pct*100:.3f}%\n"
        info_str += f"- Rebalance Frequency: every {self.rebalance_frequency} step(s)\n"
        info_str += f"{'-' * 50} PortfolioEnvironment {'-' * 50}\n"
        return info_str
