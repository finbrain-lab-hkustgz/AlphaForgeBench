from pandas_market_calendars import get_calendar
import akshare as ak
import pandas as pd
from typing import Union

from src.utils import TimeLevel

class CodeMapper():
    """
    Class to handle stock code mappings for different exchanges.
    """
    US_CODE_MAPS = {
        "New York Stock Exchange": "NYSE",
        "NASDAQ Global Select": "NASDAQ",
    }

    CHINA_CODE_MAPS = {
        "Shanghai": "XSHG",
        "Shanghai Stock Exchange": "XSHG",
        "Shenzhen": "XSHE",
        "Shenzhen Stock Exchange": "XSHE",
    }

    CRYPTO_CODE_MAPS = {
        "Binance": "BINANCE",
        "binance": "BINANCE",
        "BINANCE": "BINANCE",
    }

    @classmethod
    def get_code(cls, exchange: str) -> str:
        """
        Get the stock code for a given exchange.
        :param exchange: Name of the stock exchange.
        :return: Corresponding stock code.
        """
        if exchange in cls.US_CODE_MAPS:
            return cls.US_CODE_MAPS[exchange]
        elif exchange in cls.CHINA_CODE_MAPS:
            return cls.CHINA_CODE_MAPS[exchange]
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    @classmethod
    def get_valid_days(self,
                       exchange: str,
                       start_date: str,
                       end_date: str
                       ):
        """
        Get valid trading days for a given exchange within a date range.
        :param exchange: Name of the stock exchange.
        :param start_date: Start date for the query in 'YYYY-MM-DD' format.
        :param end_date: End date for the query in 'YYYY-MM-DD' format.
        :return: List of valid trading days as pandas DatetimeIndex.
        """
        if exchange in self.US_CODE_MAPS:
            calendar = get_calendar(self.US_CODE_MAPS[exchange])
            return calendar.valid_days(start_date=start_date, end_date=end_date)
        elif exchange in self.CHINA_CODE_MAPS:
            trade_date_df = ak.tool_trade_date_hist_sina()
            trade_days = pd.to_datetime(trade_date_df["trade_date"])
            valid_days = trade_days[(trade_days >= start_date) & (trade_days <= end_date)]
            return pd.DatetimeIndex(valid_days, dtype="datetime64[ns, UTC]")
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    def get_num_periods(self, exchange: str, level: Union[str, TimeLevel]) -> int:
        """
        Get the number of periods for a given time level.
        :param exchange: Name of the stock exchange.
        :param level: Time level as a string (e.g., '1day', '1hour') or TimeLevel enum.
        :return: Number of periods as an integer.
        """
        # If level is already a TimeLevel enum, use it directly
        if isinstance(level, TimeLevel):
            level_enum = level
        else:
            level_enum = TimeLevel.from_string(level)

        if level_enum == TimeLevel.DAY:
            if exchange in self.US_CODE_MAPS:
                return 252
            elif exchange in self.CHINA_CODE_MAPS:
                return 240
            elif exchange in self.CRYPTO_CODE_MAPS:
                return 365  # Crypto exchanges trade 24/7
        elif level_enum == TimeLevel.HOUR:
            if exchange in self.US_CODE_MAPS:
                return 252 * 6.5  # 6.5 hours per day
            elif exchange in self.CHINA_CODE_MAPS:
                return 240 * 4  # 4 hours per day
            elif exchange in self.CRYPTO_CODE_MAPS:
                return 365 * 24  # 24 hours per day, 365 days per year
        elif level_enum == TimeLevel.MINUTE:
            if exchange in self.US_CODE_MAPS:
                return 252 * 390  # 390 minutes per day
            elif exchange in self.CHINA_CODE_MAPS:
                return 240 * 240  # 240 minutes per day
            elif exchange in self.CRYPTO_CODE_MAPS:
                return 365 * 24 * 60  # 60 minutes per hour, 24 hours per day, 365 days per year
        elif level_enum == TimeLevel.SECOND:
            if exchange in self.US_CODE_MAPS:
                return 252 * 390 * 60 # 60 seconds per minute
            elif exchange in self.CHINA_CODE_MAPS:
                return 240 * 240 * 60
            elif exchange in self.CRYPTO_CODE_MAPS:
                return 365 * 24 * 60 * 60  # 60 seconds per minute, 60 minutes per hour, 24 hours per day, 365 days per year
        else:
            raise ValueError(f"Unknown time level: {level_enum} with exchange {exchange}")

    def convert_timezone(self, exchange: str, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the timestamp column of a DataFrame to the appropriate timezone based on the exchange.
        :param exchange: Name of the stock exchange.
        :param df: DataFrame containing a 'timestamp' column.
        :return: DataFrame with converted timestamps.
        """
        if exchange in self.US_CODE_MAPS:
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp']).dt.tz_convert('America/New_York')
            dataframe['timestamp'] = dataframe['timestamp'].dt.tz_localize(None)
        elif exchange in self.CHINA_CODE_MAPS:
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp']).dt.tz_convert('Asia/Shanghai')
            dataframe['timestamp'] = dataframe['timestamp'].dt.tz_localize(None)
        return dataframe

codemapper = CodeMapper()