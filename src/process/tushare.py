import os
import pandas as pd
from typing import Optional, Any
from datetime import datetime

from src.process.base import AbstractProcessor
from src.registry import PROCESSOR
from src.factor import factor_manager
from src.config import config
from src.utils import gather_with_concurrency

class TuSharePriceProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 workdir: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None

        self.workdir = workdir

    async def run(self):
        info = {
            "symbol": self.symbol,
        }

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        price_column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        data_path = os.path.join(self.data_path, "{}.jsonl".format(self.symbol))

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        assert os.path.exists(data_path), "Price path {} does not exist".format(data_path)

        df = pd.read_json(data_path, lines=True)

        df = df.rename(columns=price_column_map)[["timestamp"] + price_columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp")
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
        df = df.reset_index(drop=True)
        df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        # update df to info
        info["names"] = list(sorted(df.columns))
        info["start_date"] = df["timestamp"].min()
        info["end_date"] = df["timestamp"].max()

        df.to_json(os.path.join(self.workdir, "{}.jsonl".format(self.symbol)), orient="records", lines=True)

        return info

class TuShareFeatureProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 workdir: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None

        self.workdir = workdir

    async def run(self):

        info = {
            "symbol": self.symbol,
        }

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        price_column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        data_path = os.path.join(self.data_path, "{}.jsonl".format(self.symbol))

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        assert os.path.exists(data_path), "Price path {} does not exist".format(data_path)

        df = pd.read_json(data_path, lines=True)

        df = df.rename(columns=price_column_map)[["timestamp"] + price_columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp")
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
        df = df.reset_index(drop=True)
        df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        factors_df = await gather_with_concurrency(
            [factor_manager(factor_name, df) for factor_name in config.factors],
            max_concurrency=self.max_concurrent,
        )
        factors_df = pd.concat(factors_df, axis=1)

        info["names"] = list(sorted(factors_df.columns))
        info["start_date"] = df["timestamp"].min()
        info["end_date"] = df["timestamp"].max()

        factors_df.to_json(os.path.join(self.workdir, "{}.jsonl".format(self.symbol)), orient="records", lines=True)

        return info

class TuShareNewsProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 workdir: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent
        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.workdir = workdir

    async def run(self):
        raise NotImplementedError("TuShareNewsProcessor is not implemented yet.")
