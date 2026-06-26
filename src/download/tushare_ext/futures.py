"""
Tushare Futures Data Downloader

Downloads futures-related data:
- fut_basic: Futures contract basic information
- fut_daily: Futures daily quotes
- fut_mapping: Main contract mapping
"""

import asyncio
from typing import Optional, List, ClassVar
from datetime import datetime

import pandas as pd

from src.download.tushare_ext.base import TushareBaseDownloader
from src.logger import logger


class TushareFuturesDownloader(TushareBaseDownloader):
    """Downloader for futures data from Tushare."""

    # Supported exchanges
    EXCHANGES: ClassVar[List[str]] = ['DCE', 'SHFE', 'CZCE', 'CFFEX', 'INE']

    def __init__(
        self,
        endpoint: str = 'fut_daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        exchange: Optional[str] = None,
        fut_type: str = '1',  # 1=普通合约 2=主力 3=连续
        **kwargs
    ):
        super().__init__(
            endpoint=endpoint,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            exchange=exchange,
            **kwargs
        )
        self.fut_type = fut_type

    async def _download_fut_basic(self) -> pd.DataFrame:
        """Download futures contract basic information."""
        logger.info(f"Downloading futures basic info for exchange: {self.exchange or 'all'}...")

        all_data = []

        exchanges = [self.exchange] if self.exchange else self.EXCHANGES

        for exchange in exchanges:
            logger.info(f"Fetching fut_basic for exchange: {exchange}")
            df = await self._call_api(
                exchange=exchange,
                fut_type=self.fut_type
            )
            if len(df) > 0:
                all_data.append(df)
                logger.info(f"Got {len(df)} contracts from {exchange}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._save_data(combined_df, 'fut_basic')
            logger.info(f"Downloaded {len(combined_df)} total futures contracts")
            return combined_df

        return pd.DataFrame()

    async def _download_fut_daily_single(self, symbol: str) -> pd.DataFrame:
        """Download daily data for a single futures contract."""
        start_date = self._format_date(self.start_date)
        end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

        # Check for incremental update
        if self.incremental:
            last_date = self._get_last_date(symbol)
            if last_date:
                start_date = self._next_date(last_date)

        if start_date and end_date and int(start_date) > int(end_date):
            logger.info(f"Futures {symbol} is up to date")
            return pd.DataFrame()

        logger.info(f"Downloading fut_daily for {symbol}: {start_date} to {end_date}")

        df = await self._call_api(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if len(df) > 0:
            self._save_data(df, symbol, append=self.incremental)
            logger.info(f"Downloaded {len(df)} rows for futures {symbol}")

        return df

    async def _download_fut_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        """Download all futures data for a specific trade date."""
        logger.info(f"Downloading fut_daily for date: {trade_date}")

        params = {'trade_date': trade_date}
        if self.exchange:
            params['exchange'] = self.exchange

        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, f"daily_{trade_date}", append=False)
            logger.info(f"Downloaded {len(df)} futures records for {trade_date}")

        return df

    async def _download_fut_daily(self):
        """Download futures daily data."""
        if self.symbols:
            # Download by symbol
            logger.info(f"Downloading fut_daily for {len(self.symbols)} contracts...")
            tasks = [self._download_fut_daily_single(s) for s in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if isinstance(r, pd.DataFrame) and len(r) > 0)
            logger.info(f"Downloaded fut_daily for {success_count}/{len(self.symbols)} contracts")
        else:
            # Download by date range
            start_date = self._format_date(self.start_date) or '20200101'
            end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

            logger.info(f"Downloading fut_daily by date: {start_date} to {end_date}")

            # Generate trading dates (simplified - every weekday)
            from datetime import timedelta
            current = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')

            dates = []
            while current <= end:
                if current.weekday() < 5:  # Monday to Friday
                    dates.append(current.strftime('%Y%m%d'))
                current += timedelta(days=1)

            # Download in batches
            for i in range(0, len(dates), 10):
                batch_dates = dates[i:i+10]
                tasks = [self._download_fut_daily_by_date(d) for d in batch_dates]
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _download_fut_mapping(self):
        """Download main contract mapping."""
        if not self.symbols:
            logger.warning("No symbols specified for fut_mapping")
            return

        logger.info(f"Downloading fut_mapping for {len(self.symbols)} symbols...")

        start_date = self._format_date(self.start_date)
        end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

        for symbol in self.symbols:
            df = await self._call_api(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                self._save_data(df, f"mapping_{symbol}", append=self.incremental)
                logger.info(f"Downloaded {len(df)} mapping records for {symbol}")

    async def run(self):
        """Run the download task."""
        if self.endpoint == 'fut_basic':
            await self._download_fut_basic()
        elif self.endpoint == 'fut_daily':
            await self._download_fut_daily()
        elif self.endpoint == 'fut_mapping':
            await self._download_fut_mapping()
        else:
            raise ValueError(f"Unknown futures endpoint: {self.endpoint}")

        self._save_meta({
            'symbols': self.symbols,
            'exchange': self.exchange,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'fut_type': self.fut_type,
        })
