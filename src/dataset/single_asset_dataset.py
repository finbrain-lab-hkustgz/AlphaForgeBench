"""
Single Asset Dataset Module

This module provides a simplified dataset class for loading and processing
single asset price and feature data. The dataset concatenates price and feature
data into a single DataFrame without normalization/scaling.
"""

import os
import json
import pandas as pd
pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)

from typing import Dict, List, Any, Union
from pandas import DataFrame
from copy import deepcopy
from pydantic import BaseModel, ConfigDict

from src.utils import get_tag_name
from src.registry import DATASET
from src.utils import assemble_project_path
from src.utils import get_start_end_timestamp, TimeLevel, TimeLevelFormat


@DATASET.register_module(force=True)
class SingleAssetDataset(BaseModel):
    """
    A simplified dataset class for single asset data.
    
    This dataset loads price and optional feature data for a single symbol,
    concatenates them into a single DataFrame, and provides sliding window
    access to historical data without any normalization/scaling.
    
    Attributes:
        symbol: The trading symbol (e.g., 'BTCUSDT', 'AAPL')
        data_path: Path to the data directory containing meta_info.json and data files
        enabled_data_configs: List of data source configurations to load
        history_timestamps: Number of historical timestamps in each sample window
        start_timestamp: Start timestamp for data filtering (optional)
        end_timestamp: End timestamp for data filtering (optional)
        level: Time level granularity (e.g., '1day', '1min')
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self,
            symbol: str = None,
            data_path: str = None,
            enabled_data_configs: List[Dict[str, Any]] = None,
            history_timestamps: int = 16,
            start_timestamp: str = None,
            end_timestamp: str = None,
            level="1day",
            **kwargs
            ):
        """
        Initialize the SingleAssetDataset.
        
        Args:
            symbol: Trading symbol identifier
            data_path: Path to data directory
            enabled_data_configs: List of data source configs, each containing:
                - asset_name: Name of the asset
                - source: Data source name
                - data_type: 'price' or 'feature'
                - level: Time granularity level
            history_timestamps: Number of timestamps in each sample window
            start_timestamp: Start time for data filtering (format: 'YYYY-MM-DD HH:MM:SS')
            end_timestamp: End time for data filtering (format: 'YYYY-MM-DD HH:MM:SS')
            level: Time level string (e.g., '1day', '1min', '1hour')
        """
        super().__init__(**kwargs)

        self.symbol = symbol
        self.data_path = assemble_project_path(data_path)
        self.enabled_data_configs = enabled_data_configs
        self.history_timestamps = history_timestamps
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.level = TimeLevel.from_string(level)
        self.level_format = TimeLevelFormat.from_string(level)

        # Load metadata and parse config
        meta_info_path = os.path.join(self.data_path, "meta_info.json")
        with open(meta_info_path) as f:
            meta_info = json.load(f)
        
        # Load symbol metadata
        symbols_info = meta_info["symbols_info"]
        assert self.symbol in symbols_info, f"Symbol {self.symbol} not found in meta_info.json"
        self.symbol_metadata = symbols_info[self.symbol]
        self.symbol_info = symbols_info[self.symbol]
        
        # Parse data config
        available_tags = meta_info["tags"]
        parsed_configs = []
        for config_item in self.enabled_data_configs:
            tag = get_tag_name(
                assets_name=config_item.get("asset_name"),
                source=config_item.get("source"),
                data_type=config_item.get("data_type"),
                level=config_item.get("level")
            )
            assert tag in available_tags, f"Tag {tag} not found in meta_info.json"
            parsed_configs.append({**config_item, "tag": tag})
        
        price_configs = [item for item in parsed_configs if item["data_type"] == "price"]
        feature_configs = [item for item in parsed_configs if item["data_type"] == "feature"]
        assert len(price_configs) == 1, "Exactly one price data source must be configured."
        assert len(feature_configs) <= 1, "At most one feature data source is allowed."
        
        self.price_config = price_configs[0]
        self.feature_config = feature_configs[0] if feature_configs else None
        self.use_features = len(feature_configs) > 0

        # Load and combine data
        self.data = self._load_and_combine_data()
        self.metadata = self._build_metadata()

    def _load_and_combine_data(self) -> Dict[str, Any]:
        """
        Load price and feature data, then combine them into a single DataFrame.
        
        Returns:
            Dictionary containing:
                - symbol: Symbol identifier
                - df: DataFrame containing price and feature columns
                - price_columns: List of price column names
                - feature_columns: List of feature column names (None if not enabled)
        """
        # Load price data
        price_df = self._load_dataframe(self.price_config, self.symbol)
        price_columns = sorted(price_df.columns.tolist())
        price_df = price_df[price_columns]
        
        # Load feature data (if enabled)
        feature_df = None
        feature_columns = None
        if self.use_features and self.feature_config:
            feature_df = self._load_dataframe(self.feature_config, self.symbol)
            feature_columns = sorted(feature_df.columns.tolist())
            feature_df = feature_df[feature_columns]
        
        # Combine data
        if self.use_features and feature_df is not None:
            df = pd.concat([price_df, feature_df], axis=1, join='inner')
            # Drop rows where all features are NaN or any price is NaN
            df = df.dropna(subset=feature_columns, how='all')
            df = df.dropna(subset=price_columns)
        else:
            df = price_df
        
        return dict(
            symbol=self.symbol,
            df=df,
            price_columns=price_columns,
            feature_columns=feature_columns,
        )

    def _load_dataframe(self, config: Dict[str, Any], symbol: str) -> DataFrame:
        """
        Load DataFrame from JSONL file based on configuration.
        
        Args:
            config: Data configuration dict with 'tag' key
            symbol: Symbol identifier for file naming
            
        Returns:
            DataFrame with timestamp as index, sorted chronologically
        """
        start_time = pd.to_datetime(self.start_timestamp) if self.start_timestamp else None
        end_time = pd.to_datetime(self.end_timestamp) if self.end_timestamp else None
        
        # Load from file
        file_path = os.path.join(self.data_path, config["tag"], f"{symbol}.jsonl")
        df = pd.read_json(file_path, lines=True)
        
        # Process timestamp - handle both string format and millisecond timestamps
        ts_col = df["timestamp"]
        if ts_col.dtype == 'object':
            # String format: '2020-01-01 00:00:00'
            df["timestamp"] = pd.to_datetime(ts_col, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        elif pd.api.types.is_numeric_dtype(ts_col):
            # Numeric format: milliseconds since epoch (e.g., 1577836800000)
            df["timestamp"] = pd.to_datetime(ts_col, unit='ms', errors='coerce')
        else:
            # Try automatic parsing
            df["timestamp"] = pd.to_datetime(ts_col, errors='coerce')
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp")
        df.set_index("timestamp", inplace=True)
        
        # Filter by time range
        if start_time and end_time:
            df = df[(df.index >= start_time) & (df.index <= end_time)]
        elif start_time:
            df = df[(df.index >= start_time)]
        elif end_time:
            df = df.loc[(df.index <= end_time)]
        else:
            raise ValueError("At least one of start_timestamp or end_timestamp must be provided.")

        # Augment with extended data (index, macro, hsgt, etc.) if available
        try:
            from src.dataset.extended_data import augment_with_extended_data
            df = augment_with_extended_data(df, symbol=symbol)
        except Exception:
            pass  # Extended data augmentation failed or not available, continue with original df

        return df

    def _build_metadata(self) -> Dict[str, Any]:
        """
        Build dataset metadata including sample indices and window information.
        
        Returns:
            Dictionary containing symbol, asset_info, shape, columns, items, and length
        """
        df = self.data["df"]
        samples = {}
        
        for i in range(self.history_timestamps, len(df)):
            sample_id = len(samples)
            window_df = df.iloc[i - self.history_timestamps: i]
            
            start_time, end_time = get_start_end_timestamp(
                start_timestamp=window_df.index[0],
                end_timestamp=window_df.index[-1],
                level=self.level
            )
            
            samples[sample_id] = {
                "window_info": {
                    "start_timestamp": start_time,
                    "end_timestamp": end_time,
                    "start_index": i - self.history_timestamps,
                    "end_index": i - 1,
                }
            }
        
        return dict(
            symbol=self.symbol,
            asset_info=self.symbol_metadata,
            shape={"df": df.shape},
            columns={
                "price_columns": self.data["price_columns"],
                "feature_columns": self.data["feature_columns"] if self.use_features else None,
                "all_columns": list(df.columns),
            },
            items=samples,
            length=len(samples)
        )

    def filter_by_timestamp(self, start_timestamp: str = None, end_timestamp: str = None):
        """
        Filter dataset to include only samples within the specified time range.
        
        Args:
            start_timestamp: Start timestamp for filtering (format: 'YYYY-MM-DD HH:MM:SS')
            end_timestamp: End timestamp for filtering (format: 'YYYY-MM-DD HH:MM:SS')
        """
        start_time = pd.to_datetime(start_timestamp)
        end_time = pd.to_datetime(end_timestamp)
        
        # Find valid samples
        valid_samples = {}
        for sample_id, sample_info in self.metadata['items'].items():
            window_end = sample_info["window_info"]["end_timestamp"]
            if window_end >= start_time and window_end <= end_time:
                valid_samples[sample_id] = sample_info
        
        if not valid_samples:
            raise ValueError("No samples found in the specified time range.")
        
        # Filter data
        earliest_start = min(s["window_info"]["start_timestamp"] for s in valid_samples.values())
        df_copy = deepcopy(self.data["df"])
        filtered_df = df_copy[(df_copy.index >= earliest_start) & (df_copy.index <= end_time)]
        
        self.data["df"] = filtered_df
        self.metadata = self._build_metadata()

    def __getitem__(self, idx: int) -> DataFrame:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            DataFrame containing price and feature columns for the sample window.
        """
        window_info = self.metadata['items'][idx]["window_info"]
        start_time = window_info["start_timestamp"]
        end_time = window_info["end_timestamp"]
        
        df = self.data["df"]
        return df[(df.index >= start_time) & (df.index <= end_time)]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.metadata["length"]

    def __str__(self):
        """Return string representation of the dataset."""
        info_str = f"{'-' * 50} SingleAssetDataset {'-' * 50}\n"
        info_str += f"- Symbol: {self.symbol}\n"
        info_str += f"- Length: {self.metadata['length']}\n"
        
        shape_str = "\n".join([f"\t{key}: {value}" for key, value in self.metadata["shape"].items()])
        info_str += f"- Shape: \n {shape_str}\n"
        
        columns_str = "\n".join([f"\t{key}: {value}" for key, value in self.metadata["columns"].items()])
        info_str += f"- Columns: \n{columns_str}\n"
        
        info_str += f"{'-' * 50} SingleAssetDataset {'-' * 50}\n"
        return info_str

__all__ = [
    "SingleAssetDataset"
]
