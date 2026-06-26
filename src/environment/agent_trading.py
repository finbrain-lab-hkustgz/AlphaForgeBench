import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple

import gym
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.registry import ENVIRONMENT
from src.utils import (
    get_start_end_timestamp,
    TimeLevel,
    TradingRecords,
    Record,
    get_token_count,
    extract_boxed_content
)
from src.logger import logger

warnings.filterwarnings('ignore')

__all__ = ['AgentTradingEnvironment']


def convert_dataframe_to_markdown(
    price: pd.DataFrame,
    record: pd.DataFrame,
    valid_action: pd.DataFrame,
):
    """
    Convert dataframes to markdown strings for the prompt.
    """
    price_string = price.to_markdown(index=True)

    record_string = record.to_markdown(index=False) if not record.empty else "No records yet"
    
    note_string  = "1. `timestamp`: the timestamp of the record\n"
    note_string += "2. `open`: Open price\n"
    note_string += "3. `high`: High price\n"
    note_string += "4. `low`: Low price\n"
    note_string += "5. `close`: Close price\n"
    note_string += "6. `volume`: Volume of the asset traded\n"
    note_string += "7. `price`: Current price (adj_close price)\n"
    note_string += "8. `cash`: Current cash\n"
    note_string += "9. `position`: Current position\n"
    note_string += "10. `pre_value`: Previous total value, `value = cash + position * price`\n"
    note_string += "11. `action`: Action taken, `BUY`, `SELL`, or `HOLD`\n"
    note_string += "12. `post_value`: Current total value\n"
    note_string += "13. `ret`: Return, `ret = (post_value - pre_value) / pre_value`\n"

    valid_action_string = valid_action.to_markdown(index=False) if not valid_action.empty else "No valid actions yet"

    res_strings = dict(
        price=price_string,
        record=record_string,
        note=note_string,
        valid_action=valid_action_string,
    )

    return res_strings


@ENVIRONMENT.register_module(force=True)
class AgentTradingEnvironment(gym.Env):
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
        record_max_len: int = 32,
        valid_action_max_len: int = 8,
        use_features: bool = False,
        **kwargs,
    ):
        """
        Initialize the Agent Trading environment.
        
        Args:
            dataset: SingleAssetDataset instance
            initial_amount: Initial cash amount
            transaction_cost_pct: Transaction cost percentage
            history_timestamps: Number of historical timestamps in each sample
            step_timestamps: Number of timestamps to step forward
            start_timestamp: Start timestamp for filtering data
            end_timestamp: End timestamp for filtering data
            record_max_len: Maximum length of trading records to show in prompt
            valid_action_max_len: Maximum length of non-HOLD actions to show in prompt
            use_features: Whether to include feature columns in the agent prompt
        """
        super(AgentTradingEnvironment, self).__init__()

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
        self.record_max_len = record_max_len
        self.valid_action_max_len = valid_action_max_len
        self.use_features = use_features

        # Initialize data structures following trading.py style
        self._init_data()
        
        # Action space: SELL, HOLD, BUY
        self.action_labels = ['SELL', 'HOLD', 'BUY'] 
        self.action_dim = len(self.action_labels)

        self.record_df = pd.DataFrame() # record the trading history
        self.valid_action_df = pd.DataFrame() # record the valid action

    def _init_data(self):
        """Initialize data from the dataset following trading.py style."""
        self.df = self.dataset.data["df"]
        self.all_columns = self.dataset.metadata["columns"]["all_columns"]
        self.price_columns = self.dataset.metadata["columns"]["price_columns"]
        
        # Build timestamp info
        timestamp_info = {}
        for sample_id, sample_data in self.dataset.metadata['items'].items():
            window_info = sample_data["window_info"]
            start_ts = window_info["start_timestamp"]
            end_ts = window_info["end_timestamp"]
            
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

    def _get_dataitem(self, df: DataFrame, start_timestamp: Any, end_timestamp: Any) -> DataFrame:
        """Extract DataFrame slice within timestamp range."""
        df_copy = deepcopy(df)
        return df_copy[(df_copy.index >= start_timestamp) & (df_copy.index <= end_timestamp)]

    def get_timestamp_string(self, timestamp_index: int):
        """Get formatted timestamp string."""
        end_timestamp = self.timestamp_info[timestamp_index]["end_timestamp"]
        return end_timestamp.strftime(self.level_format.value)

    def get_price(self, timestamp_index: int) -> float:
        """Get closing price at timestamp index."""
        ts_info = self.timestamp_info[timestamp_index]
        df = self._get_dataitem(self.df, ts_info["start_timestamp"], ts_info["end_timestamp"])
        return float(df.iloc[-1]["close"])

    def get_price_full(self, timestamp_index: int) -> Tuple[float, float, float, float, float]:
        """Get all price components at timestamp index."""
        ts_info = self.timestamp_info[timestamp_index]
        df = self._get_dataitem(self.df, ts_info["start_timestamp"], ts_info["end_timestamp"])
        prices = df[self.price_columns].iloc[-1]
        return (
            float(prices["close"]),
            float(prices["high"]),
            float(prices["low"]),
            float(prices["open"]),
            float(prices["volume"])
        )

    def get_value(self, cash: float, position: float, price: float):
        """Calculate total portfolio value."""
        return cash + position * price

    def get_state(self, timestamp_index: int) -> Dict[str, Any]:
        """Generate state with markdown prompt."""
        ts_info = self.timestamp_info[timestamp_index]
        start_ts = ts_info['start_timestamp']
        end_ts = ts_info['end_timestamp']

        price_df = self._get_dataitem(self.df, start_ts, end_ts)
        if not self.use_features:
            price_df = price_df[self.price_columns]
        
        # Convert to markdown strings
        strings = convert_dataframe_to_markdown(
            price=price_df,
            record=self.record_df,
            valid_action=self.valid_action_df,
        )

        prompt = f"""
<basic_info>
This section provides basic information about the asset you are trading.
- Symbol: {self.symbol}
- Company Name: {self.symbol_info.get('companyName', self.symbol)}
- Trading Level: {self.level} (The granularity of each data row)
</basic_info>

<price_data>
This section contains historical market data for the asset. It includes standard OHLCV (Open, High, Low, Close, Volume) data. 
{"If technical indicators are present, they are provided as additional features to help identify market trends and signals." if self.use_features else "Only basic price and volume data are provided."}
{strings['price']}
</price_data>

<record_data>
This section tracks your portfolio's performance and actions over the recent {self.record_max_len} time steps. 
It records the state before and after each action, including cash, position, and total portfolio value.
{strings['record']}
</record_data>

<valid_action_data>
This section highlights your most recent {self.valid_action_max_len} 'valid' actions (BUY or SELL). 
It excludes 'HOLD' actions to provide a clear view of your active trading decisions and their outcomes.
{strings['valid_action']}
</valid_action_data>

<note>
Below are the definitions for the columns used in the tables above:
{strings['note']}
</note>

<current_info>
**IMPORTANT: YOUR CURRENT STATUS**
Today's date and time: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}
Current Asset Price: {self.price:.2f}
Available Cash: {self.cash:.2f}
Current Position (Units): {self.position:.4f}
Total Portfolio Value: {self.get_value(self.cash, self.position, self.price):.2f}
</current_info>
"""
        
        
        prompt_token_nums = get_token_count(prompt)

        state = dict(
            timestamp=end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            data=dict(
                prompt=prompt,
                prompt_token_nums=prompt_token_nums,
                symbol=self.symbol,
                asset_info=self.symbol_info
            )
        )
        return state

    def eval_buy_position(self, cash: float, price: float):
        """Evaluate maximum buyable position."""
        return cash / price / (1 + self.transaction_cost_pct)

    def eval_sell_position(self, position: float):
        """Evaluate maximum sellable position."""
        return float(position)

    def buy(self, cash: float, position: float, price: float, amount: float):
        """Execute buy action."""
        eval_buy_pos = self.eval_buy_position(cash, price)
        amount = max(0.0, min(1.0, abs(amount)))
        buy_pos = amount * eval_buy_pos

        cash = cash - (buy_pos * price * (1 + self.transaction_cost_pct))
        position = position + buy_pos
        value = self.get_value(cash, position, price)

        if buy_pos <= 1e-10:
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "BUY"
            action = self.action_labels.index("BUY")

        return {"cash": cash, "position": position, "value": value, "action": action, "action_label": action_label}

    def sell(self, cash: float, position: float, price: float, amount: float):
        """Execute sell action."""
        eval_sell_pos = self.eval_sell_position(position)
        amount = max(0.0, min(1.0, abs(amount)))
        sell_pos = amount * eval_sell_pos

        cash = cash + (sell_pos * price * (1 - self.transaction_cost_pct))
        position = position - sell_pos
        value = self.get_value(cash, position, price)

        if sell_pos <= 1e-10:
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "SELL"
            action = self.action_labels.index("SELL")

        return {"cash": cash, "position": position, "value": value, "action": action, "action_label": action_label}

    def hold(self, cash: float, position: float, price: float, amount: float = 0.0):
        """Execute hold action."""
        value = self.get_value(cash, position, price)
        return {"cash": cash, "position": position, "value": value, "action": self.action_labels.index("HOLD"), "action_label": "HOLD"}

    def _init_record(self):
        """Initialize trading records from historical data."""
        ts_info = self.timestamp_info[self.timestamp_index]
        price_df = self._get_dataitem(self.df, ts_info['start_timestamp'], ts_info['end_timestamp'])
        
        self.record_df = pd.DataFrame()
        
        rows = list(price_df.iterrows())
        for timestamp, row_values in rows[:-1]:
            p = row_values.to_dict()
            rec = Record(
                timestamp=timestamp.strftime(self.level_format.value),
                open=p.get('open', 0),
                high=p.get('high', 0),
                low=p.get('low', 0),
                close=p.get('close', 0),
                volume=p.get('volume', 0),
                price=self.price,
                cash=self.cash,
                position=int(self.position),
                pre_value=self.initial_amount,
                action='HOLD',
                post_value=self.initial_amount,
                ret=0.0,
            )
            self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(rec)])], ignore_index=True)

        # Last record placeholder for current timestamp
        timestamp, row_values = rows[-1]
        p = row_values.to_dict()
        self.last_record = Record(
            timestamp=timestamp.strftime(self.level_format.value),
            open=p.get('open', 0),
            high=p.get('high', 0),
            low=p.get('low', 0),
            close=p.get('close', 0),
            volume=p.get('volume', 0),
            price=self.price,
            cash=self.cash,
            position=int(self.position),
            pre_value=None,
            action=None,
            post_value=None,
            ret=None,
        )
        self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(self.last_record)])], ignore_index=True)
        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']

    def _add_record(self, record: Record):
        """Add a new record and maintain max length."""
        self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(record)])], ignore_index=True)
        if len(self.record_df) > self.record_max_len:
            self.record_df = self.record_df.iloc[-self.record_max_len:]
        self._refresh_valid_actions()

    def _update_record(self, record: Record):
        """Update the last record with action results."""
        if self.record_df.empty:
            return
        idx = self.record_df.index[-1]
        for field in ['pre_value', 'action', 'post_value', 'ret']:
            self.record_df.at[idx, field] = getattr(record, field)
        self._refresh_valid_actions()

    def _refresh_valid_actions(self):
        """Refresh and truncate valid actions history."""
        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']
        if len(self.valid_action_df) > self.valid_action_max_len:
            self.valid_action_df = self.valid_action_df.iloc[-self.valid_action_max_len:]

    def reset(self, **kwargs):
        """Reset the environment state."""
        self.timestamp_index = self.timestamp_min_index
        self.timestamp_string = self.get_timestamp_string(self.timestamp_index)
        self.price = self.get_price(self.timestamp_index)

        self.ret = 0.0
        self.cash = self.initial_amount
        self.position = 0.0
        self.discount = 1.0
        self.pre_value = self.value = self.initial_amount
        self.total_return = 0.0
        self.total_profit = 0.0
        self.action = 1 # HOLD
        self.action_label = 'HOLD'
        self.done = False

        self._init_record()
        self.state = self.get_state(self.timestamp_index)

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            price=self.price,
            cash=self.cash,
            position=self.position,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )
        return self.state, info

    def step(self, action: int = 0, position_ratio: float = 1.0):
        """Execute one step in the environment."""

        if action > 0:
            res = self.buy(self.cash, self.position, self.price, position_ratio)
        elif action < 0:
            res = self.sell(self.cash, self.position, self.price, position_ratio)
        else:
            res = self.hold(self.cash, self.position, self.price)

        self.cash = res['cash']
        self.position = res['position']
        self.value = res['value']
        self.action = res['action']
        self.action_label = res['action_label']

        ret = (self.value - self.pre_value) / (self.pre_value + 1e-6)
        
        # Update record with results of the action taken at current timestamp
        self.last_record.pre_value = self.pre_value
        self.last_record.action = self.action_label
        self.last_record.post_value = self.value
        self.last_record.ret = ret
        self._update_record(self.last_record)

        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        # Move to next timestamp
        self.timestamp_index += 1
        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncted = False
        else:
            self.done = True
            self.truncted = True

        self.timestamp_string = self.get_timestamp_string(self.timestamp_index)
        self.price = self.get_price(self.timestamp_index)

        if not self.done:
            # Prepare record placeholder for the NEW timestamp
            close, high, low, open, vol = self.get_price_full(self.timestamp_index)
            self.last_record = Record(
                timestamp=self.timestamp_string,
                open=open, high=high, low=low, close=close, volume=vol,
                price=self.price, cash=self.cash, position=int(self.position),
                pre_value=None, action=None, post_value=None, ret=None
            )
            self._add_record(self.last_record)

            # Update state with new timestamp info
            self.state = self.get_state(self.timestamp_index)
        
        self.pre_value = self.value

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            price=self.price,
            cash=self.cash,
            position=self.position,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )
        return self.state, ret, self.done, self.truncted, info

