"""
Extended Data Augmentation Module

Provides functionality to augment OHLCV DataFrame with extended data:
- Index data (沪深300, 中证500, 上证综指)
- Macro data (SHIBOR, LPR, GDP, CPI, PPI, M2)
- Money flow (北向资金)
- Futures data (期货主力合约)
- Options data (50ETF期权)

Usage:
    from src.dataset.extended_data import augment_with_extended_data
    df = augment_with_extended_data(df, symbol=symbol)
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

import pandas as pd

from src.config import config
from src.logger import logger


# Module-level cache for market-wide data (loaded once)
_CACHE: Dict[str, Optional[pd.DataFrame]] = {
    "root": None,
    "index": None,
    "macro": None,
    "hsgt": None,
    "futures": None,
    "options": None,
}


def _get_extended_root() -> Optional[Path]:
    """Get extended data root directory (where tushare data is stored)."""
    # Try environment variable first
    env_root = os.getenv("EXTENDED_DATA_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)

    # Try config workdir/tushare
    try:
        workdir = Path(config.workdir) / "tushare"
        if workdir.exists():
            return workdir
    except Exception:
        pass

    # Try default ./workdir/tushare
    default = Path("./workdir/tushare")
    if default.exists():
        return default

    # Fallback to ./workdir (for compatibility)
    fallback = Path("./workdir")
    if fallback.exists():
        return fallback

    return None


def _load_jsonl(file_path: Path) -> Optional[pd.DataFrame]:
    """Load a JSONL file into DataFrame."""
    if not file_path.exists():
        return None

    try:
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return None

        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def _find_data_file(root: Path, endpoint: str, filename: str = None) -> Optional[Path]:
    """
    Find data file supporting multiple directory structures:
    1. root/<endpoint>/<filename>.jsonl (e.g., workdir/tushare/shibor/data.jsonl)
    2. root/<endpoint>.jsonl (e.g., workdir/tushare/shibor.jsonl)
    3. root/macro/<endpoint>.jsonl (grouped structure)
    """
    # Try endpoint directory with specific file
    if filename:
        path = root / endpoint / filename
        if path.exists():
            return path

    # Try endpoint directory with default file names
    endpoint_dir = root / endpoint
    if endpoint_dir.exists() and endpoint_dir.is_dir():
        for default_name in [f"{endpoint}.jsonl", "data.jsonl"]:
            path = endpoint_dir / default_name
            if path.exists():
                return path
        # Try first jsonl file in directory
        jsonl_files = list(endpoint_dir.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]

    # Try flat file
    path = root / f"{endpoint}.jsonl"
    if path.exists():
        return path

    # Try grouped macro structure
    path = root / "macro" / f"{endpoint}.jsonl"
    if path.exists():
        return path

    return None


def _normalize_timestamp(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    """Normalize timestamp column and set as index."""
    if df is None or df.empty:
        return df

    if ts_col not in df.columns:
        # Try common alternatives
        for alt in ['trade_date', 'date', 'period']:
            if alt in df.columns:
                ts_col = alt
                break
        else:
            return df

    df = df.copy()

    # Handle different date formats
    ts_series = df[ts_col].astype(str)

    # Check for quarterly format (e.g., "2023Q1", "2023q1")
    if ts_series.str.match(r'^\d{4}[Qq][1-4]$').any():
        # Convert quarterly to end-of-quarter date
        def quarter_to_date(q):
            try:
                year = int(q[:4])
                quarter = int(q[-1])
                month = quarter * 3
                return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            except Exception:
                return pd.NaT
        df[ts_col] = ts_series.apply(quarter_to_date)
    # Check for monthly format (e.g., "202312", "2023-12")
    elif ts_series.str.match(r'^\d{6}$').any():
        df[ts_col] = pd.to_datetime(ts_series, format='%Y%m', errors='coerce')
    elif ts_series.str.match(r'^\d{4}-\d{2}$').any():
        df[ts_col] = pd.to_datetime(ts_series, format='%Y-%m', errors='coerce')
    else:
        # Standard datetime parsing
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

    df = df.dropna(subset=[ts_col])
    df = df.drop_duplicates(subset=[ts_col], keep='last')
    df = df.set_index(ts_col)
    df = df.sort_index()
    return df


def _align_left(base_df: pd.DataFrame, ext_df: pd.DataFrame, ffill: bool = True) -> pd.DataFrame:
    """Align extended data to base DataFrame index using left join."""
    if ext_df is None or ext_df.empty:
        return base_df

    # Reindex to base index
    ext_aligned = ext_df.reindex(base_df.index)

    # Forward fill for low-frequency data (macro, etc.)
    if ffill:
        ext_aligned = ext_aligned.ffill()

    # Join with base
    result = base_df.join(ext_aligned, how='left')
    return result


def _load_index_data(root: Path) -> Optional[pd.DataFrame]:
    """Load index daily data and rename columns."""
    # Try multiple possible paths: root/index or root/index_daily
    for subdir in ["index", "index_daily"]:
        index_dir = root / subdir
        if index_dir.exists():
            break
    else:
        return None

    # Target indices with their codes
    target_indices = {
        "000300.SH": "000300",  # 沪深300
        "000905.SH": "000905",  # 中证500
        "000001.SH": "000001",  # 上证综指
    }

    all_data = []

    for file_code, prefix in target_indices.items():
        file_path = index_dir / f"{file_code}.jsonl"
        df = _load_jsonl(file_path)

        if df is None:
            continue

        df = _normalize_timestamp(df, 'timestamp')
        if df is None or df.empty:
            continue

        # Rename columns with prefix
        rename_map = {}
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change_pct']:
            if col in df.columns:
                rename_map[col] = f"index_{prefix}_{col}"

        df = df.rename(columns=rename_map)

        # Keep only renamed columns
        keep_cols = [c for c in df.columns if c.startswith(f"index_{prefix}_")]
        df = df[keep_cols]

        all_data.append(df)

    if not all_data:
        return None

    # Merge all index data
    result = all_data[0]
    for df in all_data[1:]:
        result = result.join(df, how='outer')

    return result


def _load_macro_data(root: Path) -> Optional[pd.DataFrame]:
    """Load macro economic data and rename columns."""
    all_data = []

    # SHIBOR (daily)
    shibor_path = _find_data_file(root, "shibor")
    if shibor_path:
        shibor_df = _load_jsonl(shibor_path)
        if shibor_df is not None:
            shibor_df = _normalize_timestamp(shibor_df, 'timestamp')
            if shibor_df is not None and not shibor_df.empty:
                rename_map = {
                    'overnight': 'macro_shibor_on',
                    'week_1': 'macro_shibor_1w',
                    'week_2': 'macro_shibor_2w',
                    'month_1': 'macro_shibor_1m',
                    'month_3': 'macro_shibor_3m',
                    'month_6': 'macro_shibor_6m',
                    'month_9': 'macro_shibor_9m',
                    'year_1': 'macro_shibor_1y',
                }
                shibor_df = shibor_df.rename(columns=rename_map)
                keep_cols = [c for c in shibor_df.columns if c.startswith('macro_shibor_')]
                if keep_cols:
                    all_data.append(shibor_df[keep_cols])

    # LPR (monthly, will be ffilled)
    lpr_path = _find_data_file(root, "lpr")
    if lpr_path:
        lpr_df = _load_jsonl(lpr_path)
        if lpr_df is not None:
            lpr_df = _normalize_timestamp(lpr_df, 'timestamp')
            if lpr_df is not None and not lpr_df.empty:
                rename_map = {
                    'lpr_1y': 'macro_lpr_1y',
                    'lpr_5y': 'macro_lpr_5y',
                }
                lpr_df = lpr_df.rename(columns=rename_map)
                keep_cols = [c for c in lpr_df.columns if c.startswith('macro_lpr_')]
                if keep_cols:
                    all_data.append(lpr_df[keep_cols])

    # GDP (quarterly)
    gdp_path = _find_data_file(root, "cn_gdp")
    if gdp_path:
        gdp_df = _load_jsonl(gdp_path)
        if gdp_df is not None:
            gdp_df = _normalize_timestamp(gdp_df, 'period')
            if gdp_df is not None and not gdp_df.empty:
                rename_map = {'gdp_yoy': 'macro_gdp_yoy'}
                gdp_df = gdp_df.rename(columns=rename_map)
                if 'macro_gdp_yoy' in gdp_df.columns:
                    all_data.append(gdp_df[['macro_gdp_yoy']])

    # CPI (monthly)
    cpi_path = _find_data_file(root, "cn_cpi")
    if cpi_path:
        cpi_df = _load_jsonl(cpi_path)
        if cpi_df is not None:
            cpi_df = _normalize_timestamp(cpi_df, 'period')
            if cpi_df is not None and not cpi_df.empty:
                rename_map = {'cpi_national_yoy': 'macro_cpi_yoy'}
                cpi_df = cpi_df.rename(columns=rename_map)
                if 'macro_cpi_yoy' in cpi_df.columns:
                    all_data.append(cpi_df[['macro_cpi_yoy']])

    # PPI (monthly)
    ppi_path = _find_data_file(root, "cn_ppi")
    if ppi_path:
        ppi_df = _load_jsonl(ppi_path)
        if ppi_df is not None:
            ppi_df = _normalize_timestamp(ppi_df, 'period')
            if ppi_df is not None and not ppi_df.empty:
                rename_map = {'ppi_yoy': 'macro_ppi_yoy'}
                ppi_df = ppi_df.rename(columns=rename_map)
                if 'macro_ppi_yoy' in ppi_df.columns:
                    all_data.append(ppi_df[['macro_ppi_yoy']])

    # M2 (monthly)
    m_path = _find_data_file(root, "cn_m")
    if m_path:
        m_df = _load_jsonl(m_path)
        if m_df is not None:
            m_df = _normalize_timestamp(m_df, 'period')
            if m_df is not None and not m_df.empty:
                rename_map = {'m2_yoy': 'macro_m2_yoy'}
                m_df = m_df.rename(columns=rename_map)
                if 'macro_m2_yoy' in m_df.columns:
                    all_data.append(m_df[['macro_m2_yoy']])

    if not all_data:
        return None

    # Merge all macro data
    result = all_data[0]
    for df in all_data[1:]:
        result = result.join(df, how='outer')

    return result


def _load_hsgt_data(root: Path) -> Optional[pd.DataFrame]:
    """Load 沪深港通 money flow data."""
    # Try to find moneyflow_hsgt data file
    hsgt_path = _find_data_file(root, "moneyflow_hsgt")
    if not hsgt_path:
        # Also try moneyflow directory
        hsgt_path = _find_data_file(root, "moneyflow", "moneyflow_hsgt.jsonl")

    if not hsgt_path:
        return None

    df = _load_jsonl(hsgt_path)

    if df is None:
        return None

    df = _normalize_timestamp(df, 'timestamp')
    if df is None or df.empty:
        return None

    rename_map = {
        'north_net': 'hsgt_north_money',
        'south_net': 'hsgt_south_money',
        'sh_to_hk': 'hsgt_hgt',
        'sz_to_hk': 'hsgt_sgt',
        'hk_to_sh': 'hsgt_ggt_sh',
        'hk_to_sz': 'hsgt_ggt_sz',
    }
    df = df.rename(columns=rename_map)

    keep_cols = [c for c in df.columns if c.startswith('hsgt_')]
    if not keep_cols:
        return None

    return df[keep_cols]


def _load_futures_data(root: Path) -> Optional[pd.DataFrame]:
    """Load futures main contract data."""
    # Try multiple possible paths: root/futures or root/fut_daily
    for subdir in ["futures", "fut_daily"]:
        futures_dir = root / subdir
        if futures_dir.exists():
            break
    else:
        return None

    # Only accept explicitly named main contract files
    valid_names = ["fut_main.jsonl", "main_contract.jsonl", "IF_main.jsonl", "IC_main.jsonl"]
    main_path = None
    for name in valid_names:
        candidate = futures_dir / name
        if candidate.exists():
            main_path = candidate
            break

    if main_path is None:
        logger.debug(f"No main contract file found in {futures_dir}, skipping futures data")
        return None

    df = _load_jsonl(main_path)
    if df is None:
        return None

    df = _normalize_timestamp(df, 'timestamp')
    if df is None or df.empty:
        return None

    rename_map = {
        'close': 'fut_main_close',
        'settle': 'fut_main_settle',
        'open_interest': 'fut_main_oi',
        'volume': 'fut_main_volume',
    }
    df = df.rename(columns=rename_map)

    keep_cols = [c for c in df.columns if c.startswith('fut_main_')]
    if not keep_cols:
        return None

    return df[keep_cols]


def _load_options_data(root: Path) -> Optional[pd.DataFrame]:
    """Load 50ETF options data."""
    # Try multiple possible paths: root/options or root/opt_daily
    for subdir in ["options", "opt_daily"]:
        options_dir = root / subdir
        if options_dir.exists():
            break
    else:
        return None

    # Only accept explicitly named 50ETF option files
    valid_names = ["opt_50etf.jsonl", "50etf_option.jsonl", "510050.jsonl"]
    opt_path = None
    for name in valid_names:
        candidate = options_dir / name
        if candidate.exists():
            opt_path = candidate
            break

    if opt_path is None:
        logger.debug(f"No 50ETF option file found in {options_dir}, skipping options data")
        return None

    df = _load_jsonl(opt_path)
    if df is None:
        return None

    df = _normalize_timestamp(df, 'timestamp')
    if df is None or df.empty:
        return None

    rename_map = {
        'close': 'opt_50etf_close',
        'open_interest': 'opt_50etf_oi',
        'volume': 'opt_50etf_volume',
    }
    df = df.rename(columns=rename_map)

    keep_cols = [c for c in df.columns if c.startswith('opt_50etf_')]
    if not keep_cols:
        return None

    return df[keep_cols]


def _get_cached_data(root: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """Get cached market-wide data, loading if necessary."""
    global _CACHE

    root_str = str(root)

    # Check if cache is valid
    if _CACHE["root"] != root_str:
        # Clear cache and reload
        _CACHE = {
            "root": root_str,
            "index": None,
            "macro": None,
            "hsgt": None,
            "futures": None,
            "options": None,
        }

    # Load data if not cached
    if _CACHE["index"] is None:
        _CACHE["index"] = _load_index_data(root)

    if _CACHE["macro"] is None:
        _CACHE["macro"] = _load_macro_data(root)

    if _CACHE["hsgt"] is None:
        _CACHE["hsgt"] = _load_hsgt_data(root)

    if _CACHE["futures"] is None:
        _CACHE["futures"] = _load_futures_data(root)

    if _CACHE["options"] is None:
        _CACHE["options"] = _load_options_data(root)

    return _CACHE


def augment_with_extended_data(
    df: pd.DataFrame,
    symbol: str,
    extended_root: Optional[str] = None,
    enable: bool = True,
    ffill: bool = True,
) -> pd.DataFrame:
    """
    Augment OHLCV DataFrame with extended data.

    Args:
        df: Base DataFrame with timestamp index
        symbol: Stock symbol (for future per-symbol data like moneyflow)
        extended_root: Optional path to extended data directory
        enable: Whether to enable augmentation (False returns original df)
        ffill: Whether to forward-fill low-frequency data

    Returns:
        DataFrame with extended data columns added, or original df if no data available
    """
    if not enable:
        return df

    if df is None or df.empty:
        return df

    # Get root directory
    if extended_root:
        root = Path(extended_root)
    else:
        root = _get_extended_root()

    if root is None or not root.exists():
        # No extended data available, return original
        return df

    # Get cached market-wide data
    cache = _get_cached_data(root)

    result = df

    # Align and join each data type
    for key in ["index", "macro", "hsgt", "futures", "options"]:
        ext_df = cache.get(key)
        if ext_df is not None and not ext_df.empty:
            # Use ffill for low-frequency data (macro), not for daily data (index)
            use_ffill = ffill and key in ["macro", "hsgt"]
            result = _align_left(result, ext_df, ffill=use_ffill)

    return result


def clear_cache():
    """Clear the module-level cache."""
    global _CACHE
    _CACHE = {
        "root": None,
        "index": None,
        "macro": None,
        "hsgt": None,
        "futures": None,
        "options": None,
    }
