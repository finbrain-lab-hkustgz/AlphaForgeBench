"""
Tushare Index Data Downloader

Downloads index-related data:
- index_basic: Index basic information
- index_daily: Index daily quotes
- index_weight: Index constituent weights
"""

import asyncio
from typing import Optional, List, ClassVar
from datetime import datetime

import pandas as pd

from src.download.tushare_ext.base import TushareBaseDownloader, ENDPOINT_SPECS
from src.logger import logger


class TushareIndexDownloader(TushareBaseDownloader):
    """Downloader for index data from Tushare."""

    # Common index codes
    COMMON_INDICES: ClassVar[List[str]] = [
        '000001.SH',  # 上证综指
        '000016.SH',  # 上证50
        '000300.SH',  # 沪深300
        '000905.SH',  # 中证500
        '000852.SH',  # 中证1000
        '399001.SZ',  # 深证成指
        '399006.SZ',  # 创业板指
        '399673.SZ',  # 创业板50
        '399975.SZ',  # 证券公司
        '399986.SZ',  # 中证银行
    ]

    def __init__(
        self,
        endpoint: str = 'index_daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        market: Optional[str] = None,  # SSE, SZSE, CSI, etc.
        use_common_indices: bool = False,
        **kwargs
    ):
        super().__init__(
            endpoint=endpoint,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            **kwargs
        )
        self.market = market
        self.use_common_indices = use_common_indices

        # Use common indices if no symbols specified and flag is set
        if not self.symbols and self.use_common_indices:
            self.symbols = self.COMMON_INDICES.copy()

    async def _download_index_basic(self) -> pd.DataFrame:
        """Download index basic information."""
        logger.info("Downloading index basic information...")

        params = {}
        if self.market:
            params['market'] = self.market

        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, 'index_basic')
            logger.info(f"Downloaded {len(df)} index basic records")

        return df

    async def _download_index_daily_single(self, symbol: str) -> pd.DataFrame:
        """Download daily data for a single index."""
        start_date = self._format_date(self.start_date)
        end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

        # Check for incremental update
        if self.incremental:
            last_date = self._get_last_date(symbol)
            if last_date:
                # Start from the day after last downloaded date
                start_date = self._next_date(last_date)

        if start_date and end_date and int(start_date) > int(end_date):
            logger.info(f"Index {symbol} is up to date")
            return pd.DataFrame()

        logger.info(f"Downloading index_daily for {symbol}: {start_date} to {end_date}")

        df = await self._call_api(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if len(df) > 0:
            self._save_data(df, symbol, append=self.incremental)
            logger.info(f"Downloaded {len(df)} rows for index {symbol}")

        return df

    async def _download_index_daily(self):
        """Download index daily data for all symbols."""
        if not self.symbols:
            logger.warning("No symbols specified for index_daily. Use symbols parameter or set use_common_indices=True")
            return

        logger.info(f"Downloading index_daily for {len(self.symbols)} indices...")

        tasks = []
        for symbol in self.symbols:
            tasks.append(self._download_index_daily_single(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, pd.DataFrame) and len(r) > 0)
        logger.info(f"Downloaded index_daily for {success_count}/{len(self.symbols)} indices")

    async def _download_index_weight_single(self, index_code: str) -> pd.DataFrame:
        """Download constituent weights for a single index."""
        start_date = self._format_date(self.start_date)
        end_date = self._format_date(self.end_date) or datetime.now().strftime('%Y%m%d')

        logger.info(f"Downloading index_weight for {index_code}: {start_date} to {end_date}")

        df = await self._call_api(
            index_code=index_code,
            start_date=start_date,
            end_date=end_date
        )

        if len(df) > 0:
            self._save_data(df, f"weight_{index_code}", append=self.incremental)
            logger.info(f"Downloaded {len(df)} weight records for index {index_code}")

        return df

    async def _download_index_weight(self):
        """Download index weight data for all symbols."""
        if not self.symbols:
            # Default to major indices
            self.symbols = ['000300.SH', '000905.SH', '000852.SH']

        logger.info(f"Downloading index_weight for {len(self.symbols)} indices...")

        tasks = []
        for symbol in self.symbols:
            tasks.append(self._download_index_weight_single(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, pd.DataFrame) and len(r) > 0)
        logger.info(f"Downloaded index_weight for {success_count}/{len(self.symbols)} indices")

    async def run(self):
        """Run the download task."""
        if self.endpoint == 'index_basic':
            await self._download_index_basic()
        elif self.endpoint == 'index_daily':
            await self._download_index_daily()
        elif self.endpoint == 'index_weight':
            await self._download_index_weight()
        else:
            raise ValueError(f"Unknown index endpoint: {self.endpoint}")

        # Save metadata
        self._save_meta({
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'market': self.market,
        })
