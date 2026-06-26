"""
Tushare Macro Economic Data Downloader

Downloads macro economic data:
- cn_gdp: China GDP data
- cn_cpi: China CPI data
- cn_ppi: China PPI data
- cn_m: Money supply (M0, M1, M2)
- shibor: Shanghai Interbank Offered Rate
- lpr: Loan Prime Rate
"""

from typing import Optional, List, ClassVar
from datetime import datetime

import pandas as pd

from src.download.tushare_ext.base import TushareBaseDownloader
from src.logger import logger


class TushareMacroDownloader(TushareBaseDownloader):
    """Downloader for macro economic data from Tushare."""

    # Supported macro endpoints
    MACRO_ENDPOINTS: ClassVar[List[str]] = ['cn_gdp', 'cn_cpi', 'cn_ppi', 'cn_m', 'shibor', 'lpr']

    def __init__(
        self,
        endpoint: str = 'shibor',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_m: Optional[str] = None,  # For monthly data (YYYYMM)
        end_m: Optional[str] = None,
        start_q: Optional[str] = None,  # For quarterly data (YYYYQ1)
        end_q: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            endpoint=endpoint,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        self.start_m = start_m
        self.end_m = end_m
        self.start_q = start_q
        self.end_q = end_q

    def _get_date_params(self) -> dict:
        """Get appropriate date parameters based on endpoint."""
        params = {}

        if self.endpoint in ['cn_gdp']:
            # Quarterly data
            if self.start_q:
                params['start_q'] = self.start_q
            if self.end_q:
                params['end_q'] = self.end_q
        elif self.endpoint in ['cn_cpi', 'cn_ppi', 'cn_m']:
            # Monthly data
            if self.start_m:
                params['start_m'] = self.start_m
            elif self.start_date:
                params['start_m'] = self._format_date(self.start_date)[:6]
            if self.end_m:
                params['end_m'] = self.end_m
            elif self.end_date:
                params['end_m'] = self._format_date(self.end_date)[:6]
        elif self.endpoint in ['shibor', 'lpr']:
            # Daily data
            if self.start_date:
                params['start_date'] = self._format_date(self.start_date)
            if self.end_date:
                params['end_date'] = self._format_date(self.end_date)

        return params

    async def _download_gdp(self) -> pd.DataFrame:
        """Download GDP data."""
        logger.info("Downloading China GDP data...")

        params = self._get_date_params()
        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, 'cn_gdp')
            logger.info(f"Downloaded {len(df)} GDP records")

        return df

    async def _download_cpi(self) -> pd.DataFrame:
        """Download CPI data."""
        logger.info("Downloading China CPI data...")

        params = self._get_date_params()
        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, 'cn_cpi')
            logger.info(f"Downloaded {len(df)} CPI records")

        return df

    async def _download_ppi(self) -> pd.DataFrame:
        """Download PPI data."""
        logger.info("Downloading China PPI data...")

        params = self._get_date_params()
        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, 'cn_ppi')
            logger.info(f"Downloaded {len(df)} PPI records")

        return df

    async def _download_money_supply(self) -> pd.DataFrame:
        """Download money supply (M0, M1, M2) data."""
        logger.info("Downloading China money supply data...")

        params = self._get_date_params()
        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, 'cn_m')
            logger.info(f"Downloaded {len(df)} money supply records")

        return df

    async def _download_shibor(self) -> pd.DataFrame:
        """Download SHIBOR data."""
        logger.info("Downloading SHIBOR data...")

        params = self._get_date_params()
        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, 'shibor', append=self.incremental)
            logger.info(f"Downloaded {len(df)} SHIBOR records")

        return df

    async def _download_lpr(self) -> pd.DataFrame:
        """Download LPR data."""
        logger.info("Downloading LPR data...")

        params = self._get_date_params()
        df = await self._call_api(**params)

        if len(df) > 0:
            self._save_data(df, 'lpr', append=self.incremental)
            logger.info(f"Downloaded {len(df)} LPR records")

        return df

    async def run(self):
        """Run the download task."""
        if self.endpoint == 'cn_gdp':
            await self._download_gdp()
        elif self.endpoint == 'cn_cpi':
            await self._download_cpi()
        elif self.endpoint == 'cn_ppi':
            await self._download_ppi()
        elif self.endpoint == 'cn_m':
            await self._download_money_supply()
        elif self.endpoint == 'shibor':
            await self._download_shibor()
        elif self.endpoint == 'lpr':
            await self._download_lpr()
        else:
            raise ValueError(f"Unknown macro endpoint: {self.endpoint}")

        self._save_meta({
            'start_date': self.start_date,
            'end_date': self.end_date,
            'start_m': self.start_m,
            'end_m': self.end_m,
            'start_q': self.start_q,
            'end_q': self.end_q,
        })

    @classmethod
    async def download_all_macro(
        cls,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ):
        """Download all macro data types."""
        logger.info("Downloading all macro economic data...")

        results = {}
        for endpoint in cls.MACRO_ENDPOINTS:
            try:
                downloader = cls(
                    endpoint=endpoint,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                await downloader.run()
                results[endpoint] = 'success'
            except Exception as e:
                logger.error(f"Failed to download {endpoint}: {e}")
                results[endpoint] = f'failed: {e}'

        return results
