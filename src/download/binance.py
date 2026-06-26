import asyncio
import os
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Any

from dotenv import load_dotenv
load_dotenv(verbose=True)

try:
    from binance.spot import Spot
except ImportError:
    try:
        from binance import Spot
    except ImportError:
        raise ImportError("binance-connector not installed. Install with: pip install binance-connector")

from src.download.type import AbstractDownloader
from src.logger import logger
from src.utils import generate_intervals


class BinancePriceDownloader(AbstractDownloader):
    """Binance cryptocurrency price data downloader.
    
    Supports downloading OHLCV data for cryptocurrencies from Binance.
    Supports minute-level (1m, 5m, 15m, 30m, 1h) and day-level (1d) intervals.
    
    Reference implementation: tests/download_binance_ohlcv.py
    """
    
    def __init__(self,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent or 5

        # Map level to Binance interval
        self.binance_interval_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
            "1day": "1d",
        }
        
        if level not in self.binance_interval_map:
            raise ValueError(f"Unsupported level: {level}. Supported levels: {list(self.binance_interval_map.keys())}")
        
        self.binance_interval = self.binance_interval_map[level]
        
        # Set interval level for generate_intervals
        # Reference: FMP downloader logic
        if "day" in level:
            self.interval_level = "year"  # For day-level data, download by year
        elif "min" in level or "hour" in level:
            self.interval_level = "day"  # For minute/hour data, download day by day
        
        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)
        
        # Initialize Binance client
        self.client = Spot()

    def _check_download(self,
                        symbol: Optional[str] = None,
                        intervals: Optional[List[Tuple[datetime, datetime]]] = None
                        ):
        """Check which intervals have already been downloaded.
        
        Args:
            symbol: Trading pair symbol
            intervals: List of (start, end) datetime tuples
            
        Returns:
            List of download info dictionaries
        """
        download_infos = []

        for (start, end) in intervals:
            name = f"{start.strftime('%Y-%m-%d')}.jsonl"
            file_path = os.path.join(self.exp_path, symbol, name)
            if os.path.exists(file_path):
                item = {
                    "name": name,
                    "downloaded": True,
                    "start": start,
                    "end": end
                }
            else:
                item = {
                    "name": name,
                    "downloaded": False,
                    "start": start,
                    "end": end
                }
            download_infos.append(item)

        downloaded_items_num = len([info for info in download_infos if info["downloaded"]])
        total_items_num = len(download_infos)

        logger.info(f"| {symbol} Downloaded / Total: [{downloaded_items_num} / {total_items_num}]")

        return download_infos

    def _download_klines(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[list]:
        """Download klines (OHLCV) data from Binance.
        
        Reference: tests/download_binance_ohlcv.py download_klines function
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum number of klines to fetch per request (max 1000)
        
        Returns:
            List of kline lists
        """
        # Convert datetime to milliseconds timestamp
        end_time_ms = int(end_time.timestamp() * 1000)
        start_time_ms = int(start_time.timestamp() * 1000)
        
        all_klines = []
        current_start_time = start_time_ms
        
        # Use a set to track downloaded timestamps to avoid duplicates
        seen_timestamps = set()
        
        while True:
            try:
                # Fetch klines
                params = {
                    'symbol': symbol,
                    'interval': self.binance_interval,
                    'limit': limit
                }
                
                # Binance returns klines in chronological order (oldest first)
                # Use startTime for pagination
                if current_start_time:
                    params['startTime'] = current_start_time
                
                # Also set endTime to limit the range
                params['endTime'] = end_time_ms
                
                klines = self.client.klines(**params)
                
                if not klines:
                    break
                
                # Filter out duplicates and add new klines
                new_klines = []
                for kline in klines:
                    kline_time = kline[0]  # open_time in ms
                    if kline_time not in seen_timestamps:
                        seen_timestamps.add(kline_time)
                        new_klines.append(kline)
                
                if not new_klines:
                    # No new data, we're done
                    break
                
                all_klines.extend(new_klines)
                
                # Get the newest kline's close time (last element in the list)
                newest_kline = new_klines[-1]
                newest_close_time = newest_kline[6]  # close_time in ms
                
                # If we've reached or passed the end_time, stop
                if newest_close_time >= end_time_ms:
                    # Filter to only include klines before end_time
                    all_klines = [k for k in all_klines if k[0] < end_time_ms]
                    break
                
                # If we got fewer than limit, we've reached the end
                if len(klines) < limit:
                    break
                
                # Set next start_time to be just after the newest kline's close time
                current_start_time = newest_close_time + 1
                
            except Exception as e:
                logger.error(f"| Error downloading klines for {symbol}: {e}")
                break
        
        # Final filter by time range
        if start_time_ms:
            all_klines = [k for k in all_klines if k[0] >= start_time_ms]
        if end_time_ms:
            all_klines = [k for k in all_klines if k[0] < end_time_ms]
        
        # Sort by open_time to ensure chronological order
        all_klines.sort(key=lambda x: x[0])
        
        return all_klines

    def _klines_to_dataframe(self, klines: List[list], columns: List[str]) -> pd.DataFrame:
        """Convert Binance klines format to pandas DataFrame.
        
        Reference: tests/download_binance_ohlcv.py klines_to_dataframe function
        
        Binance kline format:
        [
            open_time (ms),
            open (str),
            high (str),
            low (str),
            close (str),
            volume (str),
            close_time (ms),
            quote_volume (str),
            trades (int),
            taker_buy_base_volume (str),
            taker_buy_quote_volume (str),
            ignore
        ]
        
        Args:
            klines: List of kline lists from Binance API
            columns: List of column names to include in DataFrame
        
        Returns:
            DataFrame with specified columns
        """
        if not klines:
            return pd.DataFrame()
        
        data = {column: [] for column in columns}
        
        for kline in klines:
            # Convert open_time from milliseconds to UTC datetime
            # Binance API returns UTC timestamps, use utcfromtimestamp to keep UTC time
            timestamp = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
            data["timestamp"].append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            data["open"].append(float(kline[1]))
            data["high"].append(float(kline[2]))
            data["low"].append(float(kline[3]))
            data["close"].append(float(kline[4]))
            data["volume"].append(float(kline[5]))
            
            # For day-level data, add change and changePercent (will be recalculated after merging)
            if "day" in self.level:
                if "change" in columns:
                    data["change"].append(0)  # Placeholder
                if "changePercent" in columns:
                    data["changePercent"].append(0)  # Placeholder
        
        df = pd.DataFrame(data, index=range(len(data["timestamp"])))
        return df

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format the DataFrame to ensure consistent column order and types.
        
        Args:
            df: DataFrame to format
            
        Returns:
            Formatted DataFrame
        """
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by="timestamp", ascending=True)
            df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            df = df[["timestamp"] + [col for col in df.columns if col != "timestamp"]]
        return df

    async def run_task(self, task: Any):
        """Run a single download task.
        
        Args:
            task: Dictionary containing task details with keys:
                - symbol: Trading pair symbol
                - start: Start datetime
                - end: End datetime
                - save_path: Path to save the data
                - columns: List of column names
                
        Returns:
            None
        """
        symbol = task["symbol"]
        start = task["start"]
        end = task["end"]
        save_path = task["save_path"]
        columns = task["columns"]

        try:
            # Adjust end time: if start and end are the same day, extend end to end of day
            # This handles the case where generate_intervals returns same-day intervals
            if start.date() == end.date():
                from datetime import timedelta
                end = start + timedelta(days=1) - timedelta(seconds=1)
            
            # Download klines for this interval
            klines = self._download_klines(symbol, start, end)
            
            if len(klines) == 0:
                logger.warning(f"| No data found for {symbol} from {start} to {end}")
                return
            
            # Convert klines to DataFrame
            df = self._klines_to_dataframe(klines, columns)
            
            if len(df) == 0:
                logger.warning(f"| No data converted to DataFrame for {symbol} from {start} to {end}")
                return
            
            # Format DataFrame
            df = self._format_dataframe(df)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_json(save_path, orient="records", lines=True)
            
            logger.info(f"| Downloaded Success: {save_path} ({len(df)} records)")
            
        except Exception as e:
            logger.error(f"| Download Failed: {save_path}, Error: {e}")

    async def run(self,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  symbol_info: Optional[Any] = None,
                  ):
        """Run the downloader to fetch data for the specified date range.
        
        Reference: tests/download_binance_ohlcv.py main function logic
        But keeps the day-by-day splitting logic from FMP downloader.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            symbol_info: Symbol information dictionary
            
        Returns:
            None
        """
        start_date = datetime.strptime(start_date if start_date
                                       else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date
                                     else self.end_date, "%Y-%m-%d")
        
        # For day-level data, extend end_date to include the full day
        # generate_intervals with right_closed=False will end at end_date - 1 day
        # So we add 1 day to end_date to ensure we include data up to the specified end_date
        if "day" in self.level:
            from datetime import timedelta
            end_date = end_date + timedelta(days=1)

        symbol_info = symbol_info if symbol_info else self.symbol_info
        symbol = symbol_info["symbol"]

        intervals = generate_intervals(start_date, end_date, self.interval_level)
        
        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )

        save_dir = os.path.join(self.exp_path, symbol)
        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for info in download_infos:
            name = info["name"]
            downloaded = info["downloaded"]
            start = info["start"]
            end = info["end"]

            if not downloaded:
                # Set columns based on level
                if "day" in self.level:
                    columns = ["timestamp", "open", "high", "low", "close", "volume", "change", "changePercent"]
                else:
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]

                save_path = os.path.join(save_dir, name)

                task = {
                    "symbol": symbol,
                    "start": start,
                    "end": end,
                    "save_path": save_path,
                    "columns": columns,
                }
                tasks.append(task)

        # Download tasks with concurrency control
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])
            # Add a small delay between batches to avoid rate limiting
            if i + self.max_concurrent < len(tasks):
                await asyncio.sleep(1)

        # After all tasks are done, concatenate downloaded files
        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )
        
        df = pd.DataFrame()
        for info in download_infos:
            name = info["name"]
            downloaded = info["downloaded"]

            if downloaded:
                chunk_df = pd.read_json(os.path.join(save_dir, name), lines=True)
                df = pd.concat([df, chunk_df], axis=0)
        
        if len(df) > 0:
            # Calculate change and changePercent for day-level data after merging
            if "day" in self.level and "change" in df.columns:
                df["change"] = df["close"].diff()
                df["changePercent"] = (df["change"] / df["close"].shift(1) * 100).fillna(0)
                # First row has no previous value, set to 0
                df.loc[df.index[0], "change"] = 0
                df.loc[df.index[0], "changePercent"] = 0
            
            df = self._format_dataframe(df)
            df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(symbol)), orient="records", lines=True)
            logger.info(f"| All data for {symbol} downloaded and saved to {self.exp_path}/{symbol}.jsonl")
        else:
            logger.warning(f"| No data downloaded for {symbol}")
