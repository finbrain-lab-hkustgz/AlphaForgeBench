"""
Tushare Money Flow Data Downloader

Downloads money flow data:
- moneyflow: Individual stock money flow
- moneyflow_hsgt: Hong Kong-Shanghai/Shenzhen Stock Connect flow
"""

import asyncio
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd

from src.download.tushare_ext.base import TushareBaseDownloader
from src.logger import logger


class TushareMoneyflowDownloader(TushareBaseDownloader):
    """Downloader for money flow data from Tushare."""

    def __init__(
        self,
        endpoint: str = 'moneyflow_hsgt',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            endpoint=endpoint,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            **kwargs
        )

    async def _download_moneyflow_single(self, symbol: str) -> pd.DataFrame:
        """Download money flow for a single stock."""
        start_date = self._format_date(self.start_date)
        end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

        if self.incremental:
            last_date = self._get_last_date(symbol)
            if last_date:
                start_date = self._next_date(last_date)

        if start_date and end_date and int(start_date) > int(end_date):
            logger.info(f"Money flow for {symbol} is up to date")
            return pd.DataFrame()

        logger.info(f"Downloading moneyflow for {symbol}: {start_date} to {end_date}")

        df = await self._call_api(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if len(df) > 0:
            self._save_data(df, symbol, append=self.incremental)
            logger.info(f"Downloaded {len(df)} money flow records for {symbol}")

        return df

    async def _download_moneyflow_by_date(self, trade_date: str) -> pd.DataFrame:
        """Download money flow for all stocks on a specific date."""
        logger.info(f"Downloading moneyflow for date: {trade_date}")

        df = await self._call_api(trade_date=trade_date)

        if len(df) > 0:
            self._save_data(df, f"moneyflow_{trade_date}", append=False)
            logger.info(f"Downloaded {len(df)} money flow records for {trade_date}")

        return df

    async def _download_moneyflow(self):
        """Download individual stock money flow data."""
        if self.symbols:
            # Download by symbol
            logger.info(f"Downloading moneyflow for {len(self.symbols)} stocks...")
            tasks = [self._download_moneyflow_single(s) for s in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if isinstance(r, pd.DataFrame) and len(r) > 0)
            logger.info(f"Downloaded moneyflow for {success_count}/{len(self.symbols)} stocks")
        else:
            # Download by date
            start_date = self._format_date(self.start_date) or '20200101'
            end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

            logger.info(f"Downloading moneyflow by date: {start_date} to {end_date}")

            current = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')

            dates = []
            while current <= end:
                if current.weekday() < 5:
                    dates.append(current.strftime('%Y%m%d'))
                current += timedelta(days=1)

            for i in range(0, len(dates), 5):
                batch_dates = dates[i:i+5]
                tasks = [self._download_moneyflow_by_date(d) for d in batch_dates]
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _download_moneyflow_hsgt(self) -> pd.DataFrame:
        """Download Hong Kong-Shanghai/Shenzhen Stock Connect money flow."""
        start_date = self._format_date(self.start_date)
        end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

        if self.incremental:
            last_date = self._get_last_date('hsgt')
            if last_date:
                start_date = self._next_date(last_date)

        if start_date and end_date and int(start_date) > int(end_date):
            logger.info("HSGT money flow is up to date")
            return pd.DataFrame()

        logger.info(f"Downloading moneyflow_hsgt: {start_date} to {end_date}")

        df = await self._call_api(
            start_date=start_date,
            end_date=end_date
        )

        if len(df) > 0:
            self._save_data(df, 'hsgt', append=self.incremental)
            logger.info(f"Downloaded {len(df)} HSGT money flow records")

        return df

    async def run(self):
        """Run the download task."""
        if self.endpoint == 'moneyflow':
            await self._download_moneyflow()
        elif self.endpoint == 'moneyflow_hsgt':
            await self._download_moneyflow_hsgt()
        else:
            raise ValueError(f"Unknown moneyflow endpoint: {self.endpoint}")

        self._save_meta({
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
        })
