"""
Tushare Extended Downloader - Unified Entry Point

Provides a single entry point for downloading various types of data from Tushare.
"""

import os
from typing import Optional, List, Union

from src.download.type import AbstractDownloader
from src.download.tushare_ext.base import ENDPOINT_SPECS
from src.download.tushare_ext.index import TushareIndexDownloader
from src.download.tushare_ext.futures import TushareFuturesDownloader
from src.download.tushare_ext.options import TushareOptionsDownloader
from src.download.tushare_ext.macro import TushareMacroDownloader
from src.download.tushare_ext.moneyflow import TushareMoneyflowDownloader
from src.registry import DOWNLOADER
from src.config import config
from src.logger import logger


# Endpoint to data type mapping
ENDPOINT_TO_TYPE = {
    # Index
    'index_basic': 'index',
    'index_daily': 'index',
    'index_weight': 'index',
    # Futures
    'fut_basic': 'futures',
    'fut_daily': 'futures',
    'fut_mapping': 'futures',
    # Options
    'opt_basic': 'options',
    'opt_daily': 'options',
    # Macro
    'cn_gdp': 'macro',
    'cn_cpi': 'macro',
    'cn_ppi': 'macro',
    'cn_m': 'macro',
    'shibor': 'macro',
    'lpr': 'macro',
    # Money flow
    'moneyflow': 'moneyflow',
    'moneyflow_hsgt': 'moneyflow',
    # Daily basic - TODO: implement fundamental downloader
    # 'daily_basic': 'fundamental',
}

# Data type to downloader class mapping
TYPE_TO_DOWNLOADER = {
    'index': TushareIndexDownloader,
    'futures': TushareFuturesDownloader,
    'options': TushareOptionsDownloader,
    'macro': TushareMacroDownloader,
    'moneyflow': TushareMoneyflowDownloader,
}


@DOWNLOADER.register_module(force=True)
class TushareExtendedDownloader(AbstractDownloader):
    """
    Unified entry point for Tushare extended data downloads.

    Usage:
        # Download index daily data
        downloader = TushareExtendedDownloader(
            endpoint='index_daily',
            start_date='2020-01-01',
            end_date='2023-12-31',
            symbols=['000300.SH', '000905.SH']
        )
        await downloader.run()

        # Download macro data
        downloader = TushareExtendedDownloader(
            endpoint='shibor',
            start_date='2020-01-01'
        )
        await downloader.run()

        # Download multiple endpoints
        downloader = TushareExtendedDownloader(
            endpoints=['index_daily', 'shibor', 'moneyflow_hsgt'],
            start_date='2020-01-01'
        )
        await downloader.run()
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        endpoints: Optional[List[str]] = None,
        data_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        exchange: Optional[str] = None,
        incremental: bool = True,
        max_concurrent: int = 5,
        calls_per_minute: int = 80,
        exp_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        # Handle single endpoint or multiple endpoints
        self.endpoints = []
        if endpoint:
            self.endpoints.append(endpoint)
        if endpoints:
            self.endpoints.extend(endpoints)

        # If data_type is specified, get all endpoints for that type
        if data_type and not self.endpoints:
            self.endpoints = [ep for ep, dt in ENDPOINT_TO_TYPE.items() if dt == data_type]

        if not self.endpoints:
            raise ValueError("Must specify endpoint, endpoints, or data_type")

        # Validate endpoints
        for ep in self.endpoints:
            if ep not in ENDPOINT_SPECS:
                raise ValueError(f"Unknown endpoint: {ep}. Available: {list(ENDPOINT_SPECS.keys())}")

        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.exchange = exchange
        self.incremental = incremental
        self.max_concurrent = max_concurrent
        self.calls_per_minute = calls_per_minute

        self.workdir = config.workdir if hasattr(config, 'workdir') else 'workdir'
        self.exp_path = exp_path or os.path.join(self.workdir, 'tushare')
        os.makedirs(self.exp_path, exist_ok=True)

        self.extra_kwargs = kwargs

    def _get_downloader(self, endpoint: str) -> AbstractDownloader:
        """Get the appropriate downloader for an endpoint."""
        data_type = ENDPOINT_TO_TYPE.get(endpoint)
        if not data_type:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        downloader_class = TYPE_TO_DOWNLOADER.get(data_type)
        if not downloader_class:
            raise ValueError(f"No downloader for data type: {data_type}")

        # Create downloader with appropriate path
        exp_path = os.path.join(self.exp_path, data_type)

        return downloader_class(
            endpoint=endpoint,
            start_date=self.start_date,
            end_date=self.end_date,
            symbols=self.symbols,
            exchange=self.exchange,
            incremental=self.incremental,
            max_concurrent=self.max_concurrent,
            calls_per_minute=self.calls_per_minute,
            exp_path=exp_path,
            **self.extra_kwargs
        )

    async def run(self):
        """Run all download tasks."""
        logger.info(f"Starting Tushare extended download for endpoints: {self.endpoints}")

        results = {}
        for endpoint in self.endpoints:
            try:
                logger.info(f"Downloading {endpoint}...")
                downloader = self._get_downloader(endpoint)
                await downloader.run()
                results[endpoint] = 'success'
                logger.info(f"Completed {endpoint}")
            except Exception as e:
                logger.error(f"Failed to download {endpoint}: {e}")
                results[endpoint] = f'failed: {e}'

        # Summary
        success_count = sum(1 for v in results.values() if v == 'success')
        logger.info(f"Download complete: {success_count}/{len(self.endpoints)} endpoints succeeded")

        return results

    @classmethod
    def list_endpoints(cls) -> dict:
        """List all available endpoints grouped by data type."""
        grouped = {}
        for endpoint, data_type in ENDPOINT_TO_TYPE.items():
            if data_type not in grouped:
                grouped[data_type] = []
            spec = ENDPOINT_SPECS.get(endpoint)
            grouped[data_type].append({
                'endpoint': endpoint,
                'required_points': spec.required_points if spec else 'unknown',
            })
        return grouped

    @classmethod
    def get_endpoint_info(cls, endpoint: str) -> dict:
        """Get information about a specific endpoint."""
        spec = ENDPOINT_SPECS.get(endpoint)
        if not spec:
            return {'error': f'Unknown endpoint: {endpoint}'}

        return {
            'endpoint': endpoint,
            'data_type': ENDPOINT_TO_TYPE.get(endpoint),
            'has_symbol': spec.has_symbol,
            'has_date_range': spec.has_date_range,
            'is_static': spec.is_static,
            'required_points': spec.required_points,
            'field_mapping': spec.field_mapping,
        }
