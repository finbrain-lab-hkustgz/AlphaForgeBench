"""
Tushare Base Downloader

Provides common functionality for all Tushare downloaders:
- API client initialization with rate limiting
- Unified data saving (JSONL format)
- Error handling and retry logic
- Incremental update support
"""

import os
import json
import asyncio
from abc import abstractmethod
from typing import Optional, Any, Dict, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv(verbose=True)

from src.download.type import AbstractDownloader
from src.logger import logger
from src.config import config


@dataclass
class EndpointSpec:
    """Specification for a Tushare API endpoint."""
    name: str  # API endpoint name, e.g., 'index_daily'
    date_field: str = 'trade_date'  # Field name for date filtering
    symbol_field: str = 'ts_code'  # Field name for symbol filtering
    has_symbol: bool = True  # Whether endpoint requires symbol parameter
    has_date_range: bool = True  # Whether endpoint supports date range
    is_static: bool = False  # Whether data is static (one-time fetch)
    output_fields: List[str] = field(default_factory=list)  # Fields to keep
    field_mapping: Dict[str, str] = field(default_factory=dict)  # Field name mapping
    required_points: int = 120  # Required Tushare points
    max_rows_per_call: int = 5000  # Max rows returned per API call


# Common endpoint specifications
ENDPOINT_SPECS: Dict[str, EndpointSpec] = {
    # Index endpoints
    'index_daily': EndpointSpec(
        name='index_daily',
        date_field='trade_date',
        symbol_field='ts_code',
        has_symbol=True,
        has_date_range=True,
        field_mapping={
            'trade_date': 'timestamp',
            'ts_code': 'symbol',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'pct_chg': 'change_pct',
        },
        required_points=120,
    ),
    'index_basic': EndpointSpec(
        name='index_basic',
        has_symbol=False,
        has_date_range=False,
        is_static=True,
        field_mapping={
            'ts_code': 'symbol',
            'name': 'name',
            'market': 'market',
            'publisher': 'publisher',
            'category': 'category',
            'base_date': 'base_date',
            'base_point': 'base_point',
        },
        required_points=120,
    ),
    'index_weight': EndpointSpec(
        name='index_weight',
        date_field='trade_date',
        symbol_field='index_code',
        has_symbol=True,
        has_date_range=True,
        field_mapping={
            'index_code': 'index_symbol',
            'con_code': 'constituent_symbol',
            'trade_date': 'timestamp',
            'weight': 'weight',
        },
        required_points=400,
    ),

    # Futures endpoints
    'fut_basic': EndpointSpec(
        name='fut_basic',
        has_symbol=False,
        has_date_range=False,
        is_static=True,
        field_mapping={
            'ts_code': 'symbol',
            'symbol': 'contract_symbol',
            'exchange': 'exchange',
            'name': 'name',
            'fut_code': 'fut_code',
            'multiplier': 'multiplier',
            'trade_unit': 'trade_unit',
            'per_unit': 'per_unit',
            'list_date': 'list_date',
            'delist_date': 'delist_date',
        },
        required_points=120,
    ),
    'fut_daily': EndpointSpec(
        name='fut_daily',
        date_field='trade_date',
        symbol_field='ts_code',
        has_symbol=True,
        has_date_range=True,
        field_mapping={
            'ts_code': 'symbol',
            'trade_date': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'settle': 'settle',
            'pre_settle': 'pre_settle',
            'pre_close': 'pre_close',
            'vol': 'volume',
            'amount': 'amount',
            'oi': 'open_interest',
            'oi_chg': 'oi_change',
        },
        required_points=2000,
    ),
    'fut_mapping': EndpointSpec(
        name='fut_mapping',
        date_field='trade_date',
        symbol_field='ts_code',
        has_symbol=True,
        has_date_range=True,
        field_mapping={
            'ts_code': 'symbol',
            'trade_date': 'timestamp',
            'mapping_ts_code': 'main_contract',
        },
        required_points=2000,
    ),

    # Options endpoints
    'opt_basic': EndpointSpec(
        name='opt_basic',
        has_symbol=False,
        has_date_range=False,
        is_static=True,
        field_mapping={
            'ts_code': 'symbol',
            'exchange': 'exchange',
            'name': 'name',
            'per_unit': 'per_unit',
            'opt_code': 'opt_code',
            'opt_type': 'opt_type',
            'call_put': 'call_put',
            'exercise_type': 'exercise_type',
            'exercise_price': 'exercise_price',
            's_month': 's_month',
            'maturity_date': 'maturity_date',
            'list_date': 'list_date',
            'delist_date': 'delist_date',
        },
        required_points=120,
    ),
    'opt_daily': EndpointSpec(
        name='opt_daily',
        date_field='trade_date',
        symbol_field='ts_code',
        has_symbol=True,
        has_date_range=True,
        field_mapping={
            'ts_code': 'symbol',
            'trade_date': 'timestamp',
            'exchange': 'exchange',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'settle': 'settle',
            'pre_settle': 'pre_settle',
            'pre_close': 'pre_close',
            'vol': 'volume',
            'amount': 'amount',
            'oi': 'open_interest',
        },
        required_points=2000,
        max_rows_per_call=15000,
    ),

    # Macro endpoints
    'cn_gdp': EndpointSpec(
        name='cn_gdp',
        date_field='quarter',
        has_symbol=False,
        has_date_range=True,
        field_mapping={
            'quarter': 'period',
            'gdp': 'gdp',
            'gdp_yoy': 'gdp_yoy',
            'pi': 'primary_industry',
            'si': 'secondary_industry',
            'ti': 'tertiary_industry',
        },
        required_points=600,
    ),
    'cn_cpi': EndpointSpec(
        name='cn_cpi',
        date_field='month',
        has_symbol=False,
        has_date_range=True,
        field_mapping={
            'month': 'period',
            'nt_val': 'cpi_national',
            'nt_yoy': 'cpi_national_yoy',
            'nt_mom': 'cpi_national_mom',
            'town_val': 'cpi_urban',
            'town_yoy': 'cpi_urban_yoy',
            'cnt_val': 'cpi_rural',
            'cnt_yoy': 'cpi_rural_yoy',
        },
        required_points=600,
    ),
    'cn_ppi': EndpointSpec(
        name='cn_ppi',
        date_field='month',
        has_symbol=False,
        has_date_range=True,
        field_mapping={
            'month': 'period',
            'ppi_yoy': 'ppi_yoy',
            'ppi_mp_yoy': 'ppi_means_production_yoy',
            'ppi_cg_yoy': 'ppi_consumer_goods_yoy',
        },
        required_points=600,
    ),
    'cn_m': EndpointSpec(
        name='cn_m',
        date_field='month',
        has_symbol=False,
        has_date_range=True,
        field_mapping={
            'month': 'period',
            'm0': 'm0',
            'm0_yoy': 'm0_yoy',
            'm1': 'm1',
            'm1_yoy': 'm1_yoy',
            'm2': 'm2',
            'm2_yoy': 'm2_yoy',
        },
        required_points=600,
    ),
    'shibor': EndpointSpec(
        name='shibor',
        date_field='date',
        has_symbol=False,
        has_date_range=True,
        field_mapping={
            'date': 'timestamp',
            'on': 'overnight',
            '1w': 'week_1',
            '2w': 'week_2',
            '1m': 'month_1',
            '3m': 'month_3',
            '6m': 'month_6',
            '9m': 'month_9',
            '1y': 'year_1',
        },
        required_points=120,
    ),
    'lpr': EndpointSpec(
        name='lpr',
        date_field='date',
        has_symbol=False,
        has_date_range=True,
        field_mapping={
            'date': 'timestamp',
            'lpr1y': 'lpr_1y',
            'lpr5y': 'lpr_5y',
        },
        required_points=120,
    ),

    # Money flow endpoints
    'moneyflow': EndpointSpec(
        name='moneyflow',
        date_field='trade_date',
        symbol_field='ts_code',
        has_symbol=True,
        has_date_range=True,
        field_mapping={
            'ts_code': 'symbol',
            'trade_date': 'timestamp',
            'buy_sm_vol': 'buy_small_vol',
            'buy_sm_amount': 'buy_small_amount',
            'sell_sm_vol': 'sell_small_vol',
            'sell_sm_amount': 'sell_small_amount',
            'buy_md_vol': 'buy_medium_vol',
            'buy_md_amount': 'buy_medium_amount',
            'sell_md_vol': 'sell_medium_vol',
            'sell_md_amount': 'sell_medium_amount',
            'buy_lg_vol': 'buy_large_vol',
            'buy_lg_amount': 'buy_large_amount',
            'sell_lg_vol': 'sell_large_vol',
            'sell_lg_amount': 'sell_large_amount',
            'buy_elg_vol': 'buy_xlarge_vol',
            'buy_elg_amount': 'buy_xlarge_amount',
            'sell_elg_vol': 'sell_xlarge_vol',
            'sell_elg_amount': 'sell_xlarge_amount',
            'net_mf_vol': 'net_flow_vol',
            'net_mf_amount': 'net_flow_amount',
        },
        required_points=2000,
    ),
    'moneyflow_hsgt': EndpointSpec(
        name='moneyflow_hsgt',
        date_field='trade_date',
        has_symbol=False,
        has_date_range=True,
        field_mapping={
            'trade_date': 'timestamp',
            'ggt_ss': 'hk_to_sh',  # 港股通(沪)
            'ggt_sz': 'hk_to_sz',  # 港股通(深)
            'hgt': 'sh_to_hk',     # 沪股通
            'sgt': 'sz_to_hk',     # 深股通
            'north_money': 'north_net',  # 北向资金
            'south_money': 'south_net',  # 南向资金
        },
        required_points=2000,
    ),

    # Daily basic (valuation)
    'daily_basic': EndpointSpec(
        name='daily_basic',
        date_field='trade_date',
        symbol_field='ts_code',
        has_symbol=True,
        has_date_range=True,
        field_mapping={
            'ts_code': 'symbol',
            'trade_date': 'timestamp',
            'close': 'close',
            'turnover_rate': 'turnover_rate',
            'turnover_rate_f': 'turnover_rate_free',
            'volume_ratio': 'volume_ratio',
            'pe': 'pe',
            'pe_ttm': 'pe_ttm',
            'pb': 'pb',
            'ps': 'ps',
            'ps_ttm': 'ps_ttm',
            'dv_ratio': 'dividend_yield',
            'dv_ttm': 'dividend_yield_ttm',
            'total_share': 'total_shares',
            'float_share': 'float_shares',
            'free_share': 'free_shares',
            'total_mv': 'market_cap',
            'circ_mv': 'float_market_cap',
        },
        required_points=2000,
    ),
}


class TushareRateLimiter:
    """Rate limiter for Tushare API calls."""

    def __init__(self, calls_per_minute: int = 80, max_concurrent: int = 5):
        self.calls_per_minute = calls_per_minute
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.call_times: List[datetime] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make an API call."""
        await self.semaphore.acquire()

        async with self._lock:
            now = datetime.now()
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < timedelta(minutes=1)]

            # Wait if we've hit the rate limit
            if len(self.call_times) >= self.calls_per_minute:
                wait_time = 60 - (now - self.call_times[0]).total_seconds()
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    self.call_times = []

            self.call_times.append(datetime.now())

    def release(self):
        """Release the semaphore."""
        self.semaphore.release()


class TushareBaseDownloader(AbstractDownloader):
    """Base class for all Tushare downloaders."""

    # Shared rate limiter across all instances
    _rate_limiter: Optional[TushareRateLimiter] = None
    _pro: Optional[Any] = None

    def __init__(
        self,
        endpoint: str,
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

        self.endpoint = endpoint
        self.spec = ENDPOINT_SPECS.get(endpoint)
        if not self.spec:
            raise ValueError(f"Unknown endpoint: {endpoint}. Available: {list(ENDPOINT_SPECS.keys())}")

        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols or []
        self.exchange = exchange
        self.incremental = incremental
        self.max_concurrent = max_concurrent
        self.calls_per_minute = calls_per_minute

        # Set up paths
        self.workdir = config.workdir if hasattr(config, 'workdir') else 'workdir'
        self.exp_path = exp_path or os.path.join(self.workdir, 'tushare', endpoint)
        os.makedirs(self.exp_path, exist_ok=True)

        # Initialize shared resources
        self._init_shared_resources()

    def _init_shared_resources(self):
        """Initialize shared API client and rate limiter."""
        if TushareBaseDownloader._pro is None:
            token = os.getenv("TUSHARE_API_KEY") or os.getenv("TUSHARE_TOKEN")
            if not token:
                raise ValueError("TUSHARE_API_KEY or TUSHARE_TOKEN not found in environment")
            TushareBaseDownloader._pro = ts.pro_api(token=token)
            logger.info("Tushare API client initialized")

        if TushareBaseDownloader._rate_limiter is None:
            TushareBaseDownloader._rate_limiter = TushareRateLimiter(
                calls_per_minute=self.calls_per_minute,
                max_concurrent=self.max_concurrent
            )

    @property
    def pro(self):
        """Get the Tushare pro API client."""
        return TushareBaseDownloader._pro

    @property
    def rate_limiter(self):
        """Get the rate limiter."""
        return TushareBaseDownloader._rate_limiter

    def _format_date(self, date_str: str) -> str:
        """Format date string to Tushare format (YYYYMMDD)."""
        if not date_str:
            return None
        # Handle various formats
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y%m%d')
            except ValueError:
                continue
        return date_str

    def _next_date(self, date_str: str) -> str:
        """Get the next calendar date in YYYYMMDD format."""
        if not date_str:
            return None
        try:
            dt = datetime.strptime(date_str, '%Y%m%d')
            next_dt = dt + timedelta(days=1)
            return next_dt.strftime('%Y%m%d')
        except ValueError:
            return date_str

    def _apply_field_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply field name mapping to DataFrame."""
        if not self.spec.field_mapping:
            return df

        # Only rename columns that exist
        rename_map = {k: v for k, v in self.spec.field_mapping.items() if k in df.columns}
        return df.rename(columns=rename_map)

    def _format_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format timestamp column to standard format."""
        timestamp_col = self.spec.field_mapping.get(self.spec.date_field, self.spec.date_field)

        if timestamp_col not in df.columns:
            return df

        # Try to parse as datetime
        try:
            if df[timestamp_col].dtype == 'object':
                # Handle YYYYMMDD format
                if df[timestamp_col].str.match(r'^\d{8}$').all():
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y%m%d')
                else:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            # Format to standard string
            if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = df[timestamp_col].dt.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Failed to format timestamp: {e}")

        return df

    def _get_last_date(self, symbol: Optional[str] = None) -> Optional[str]:
        """Get the last downloaded date for incremental update."""
        if not self.incremental:
            return None

        if symbol:
            file_path = os.path.join(self.exp_path, f"{symbol}.jsonl")
        else:
            file_path = os.path.join(self.exp_path, f"{self.endpoint}.jsonl")

        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_json(file_path, lines=True)
            if len(df) == 0:
                return None

            timestamp_col = self.spec.field_mapping.get(self.spec.date_field, 'timestamp')
            if timestamp_col in df.columns:
                last_date = df[timestamp_col].max()
                if isinstance(last_date, str):
                    return last_date.replace('-', '')
                return last_date.strftime('%Y%m%d')
        except Exception as e:
            logger.warning(f"Failed to read last date: {e}")

        return None

    def _save_data(self, df: pd.DataFrame, filename: str, append: bool = False):
        """Save DataFrame to JSONL file."""
        if df is None or len(df) == 0:
            return

        file_path = os.path.join(self.exp_path, f"{filename}.jsonl")

        # Apply field mapping and format
        df = self._apply_field_mapping(df)
        df = self._format_timestamp(df)

        # Sort by timestamp if available
        timestamp_col = self.spec.field_mapping.get(self.spec.date_field, 'timestamp')
        if timestamp_col in df.columns:
            df = df.sort_values(by=timestamp_col, ascending=True)

        if append and os.path.exists(file_path):
            # Read existing data and merge
            existing_df = pd.read_json(file_path, lines=True)
            df = pd.concat([existing_df, df], ignore_index=True)

            # Remove duplicates based on key columns
            # Use all available key columns for deduplication
            potential_keys = [timestamp_col, 'symbol', 'index_symbol', 'constituent_symbol']
            key_cols = [c for c in potential_keys if c in df.columns]
            if key_cols:
                df = df.drop_duplicates(subset=key_cols, keep='last')

            if timestamp_col in df.columns:
                df = df.sort_values(by=timestamp_col, ascending=True)

        df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved {len(df)} rows to {file_path}")

    def _save_meta(self, meta: Dict[str, Any]):
        """Save metadata file."""
        meta_path = os.path.join(self.exp_path, 'meta.json')
        meta['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        meta['endpoint'] = self.endpoint

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    async def _call_api(self, **kwargs) -> pd.DataFrame:
        """Call Tushare API with rate limiting and error handling."""
        max_retries = 3
        retry_count = 0
        rate_limit_hit = False

        while retry_count < max_retries:
            await self.rate_limiter.acquire()
            rate_limit_hit = False

            try:
                # Get the API method
                api_method = getattr(self.pro, self.endpoint)

                # Make the call (Tushare is synchronous)
                df = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: api_method(**kwargs)
                )

                if df is None:
                    return pd.DataFrame()

                return df

            except Exception as e:
                error_msg = str(e)
                if '积分' in error_msg or 'point' in error_msg.lower():
                    logger.error(f"Insufficient Tushare points for {self.endpoint}: {e}")
                    return pd.DataFrame()
                elif '频率' in error_msg or 'rate' in error_msg.lower():
                    retry_count += 1
                    rate_limit_hit = True
                    logger.warning(f"Rate limit hit, waiting 60s... (retry {retry_count}/{max_retries})")
                else:
                    logger.error(f"API call failed for {self.endpoint}: {e}")
                    return pd.DataFrame()

            finally:
                # Always release semaphore
                self.rate_limiter.release()

            # Sleep outside of try/finally block so semaphore is released
            if rate_limit_hit:
                await asyncio.sleep(60)

        logger.error(f"Max retries exceeded for {self.endpoint}")
        return pd.DataFrame()

    @abstractmethod
    async def run(self):
        """Run the download task. Must be implemented by subclasses."""
        pass
