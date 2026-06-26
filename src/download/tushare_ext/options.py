"""
Tushare Options Data Downloader

Downloads options-related data:
- opt_basic: Options contract basic information
- opt_daily: Options daily quotes
"""

import asyncio
from typing import Optional, List, ClassVar
from datetime import datetime

import pandas as pd

from src.download.tushare_ext.base import TushareBaseDownloader
from src.logger import logger


class TushareOptionsDownloader(TushareBaseDownloader):
    """Downloader for options data from Tushare."""

    # Supported exchanges
    EXCHANGES: ClassVar[List[str]] = ['SSE', 'SZSE', 'CFFEX', 'DCE', 'SHFE', 'CZCE']

    def __init__(
        self,
        endpoint: str = 'opt_daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        exchange: Optional[str] = None,
        call_put: Optional[str] = None,  # C=Call, P=Put
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
        self.call_put = call_put

    async def _download_opt_basic(self) -> pd.DataFrame:
        """Download options contract basic information."""
        logger.info(f"Downloading options basic info for exchange: {self.exchange or 'all'}...")

        all_data = []

        exchanges = [self.exchange] if self.exchange else self.EXCHANGES

        for exchange in exchanges:
            logger.info(f"Fetching opt_basic for exchange: {exchange}")

            params = {'exchange': exchange}
            if self.call_put:
                params['call_put'] = self.call_put

            df = await self._call_api(**params)
            if len(df) > 0:
                all_data.append(df)
                logger.info(f"Got {len(df)} options contracts from {exchange}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._save_data(combined_df, 'opt_basic')
            logger.info(f"Downloaded {len(combined_df)} total options contracts")
            return combined_df

        return pd.DataFrame()

    async def _download_opt_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        """Download all options data for a specific trade date."""
        logger.info(f"Downloading opt_daily for date: {trade_date}")

        params = {'trade_date': trade_date}
        if self.exchange:
            params['exchange'] = self.exchange

        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, f"daily_{trade_date}", append=False)
            logger.info(f"Downloaded {len(df)} options records for {trade_date}")

        return df

    async def _download_opt_daily_single(self, symbol: str) -> pd.DataFrame:
        """Download daily data for a single options contract."""
        start_date = self._format_date(self.start_date)
        end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

        if self.incremental:
            last_date = self._get_last_date(symbol)
            if last_date:
                start_date = self._next_date(last_date)

        if start_date and end_date and int(start_date) > int(end_date):
            logger.info(f"Options {symbol} is up to date")
            return pd.DataFrame()

        logger.info(f"Downloading opt_daily for {symbol}: {start_date} to {end_date}")

        df = await self._call_api(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if len(df) > 0:
            self._save_data(df, symbol, append=self.incremental)
            logger.info(f"Downloaded {len(df)} rows for options {symbol}")

        return df

    async def _download_opt_daily(self):
        """Download options daily data."""
        if self.symbols:
            # Download by symbol
            logger.info(f"Downloading opt_daily for {len(self.symbols)} contracts...")
            tasks = [self._download_opt_daily_single(s) for s in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if isinstance(r, pd.DataFrame) and len(r) > 0)
            logger.info(f"Downloaded opt_daily for {success_count}/{len(self.symbols)} contracts")
        else:
            # Download by date - more practical for options due to large number of contracts
            start_date = self._format_date(self.start_date) or '20200101'
            end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

            logger.info(f"Downloading opt_daily by date: {start_date} to {end_date}")

            from datetime import timedelta
            current = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')

            dates = []
            while current <= end:
                if current.weekday() < 5:
                    dates.append(current.strftime('%Y%m%d'))
                current += timedelta(days=1)

            # Download in batches
            for i in range(0, len(dates), 5):
                batch_dates = dates[i:i+5]
                tasks = [self._download_opt_daily_by_date(d) for d in batch_dates]
                await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self):
        """Run the download task."""
        if self.endpoint == 'opt_basic':
            await self._download_opt_basic()
        elif self.endpoint == 'opt_daily':
            await self._download_opt_daily()
        else:
            raise ValueError(f"Unknown options endpoint: {self.endpoint}")

        self._save_meta({
            'symbols': self.symbols,
            'exchange': self.exchange,
            'call_put': self.call_put,
            'start_date': self.start_date,
            'end_date': self.end_date,
        })
