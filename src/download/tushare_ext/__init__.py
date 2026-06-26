"""
Tushare Extended Downloaders Package

This package provides extended data downloaders for Tushare API,
supporting various data types beyond basic stock prices:
- Index data (index_daily, index_basic, index_weight)
- Futures data (fut_daily, fut_basic)
- Options data (opt_daily, opt_basic)
- Macro economic data (cn_gdp, cn_cpi, cn_ppi, shibor, lpr, etc.)
- Money flow data (moneyflow, moneyflow_hsgt)
"""

from src.download.tushare_ext.base import TushareBaseDownloader
from src.download.tushare_ext.index import TushareIndexDownloader
from src.download.tushare_ext.futures import TushareFuturesDownloader
from src.download.tushare_ext.options import TushareOptionsDownloader
from src.download.tushare_ext.macro import TushareMacroDownloader
from src.download.tushare_ext.moneyflow import TushareMoneyflowDownloader
from src.download.tushare_ext.downloader import TushareExtendedDownloader

__all__ = [
    "TushareBaseDownloader",
    "TushareIndexDownloader",
    "TushareFuturesDownloader",
    "TushareOptionsDownloader",
    "TushareMacroDownloader",
    "TushareMoneyflowDownloader",
    "TushareExtendedDownloader",
]
