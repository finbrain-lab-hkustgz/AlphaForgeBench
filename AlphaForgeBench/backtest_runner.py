"""Backtest runner for validating generated strategy code."""

import asyncio
import inspect
import json
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from src.logger import logger


@dataclass
class BacktestResult:
    """Result of a single backtest."""

    query_id: str
    model: str
    sample_id: int
    syntax_valid: bool = False
    backtest_valid: bool = False
    # Aggregated Metrics (average across all symbols)
    sharpe: Optional[float] = None
    total_return: Optional[float] = None
    annual_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    sortino: Optional[float] = None
    calmar: Optional[float] = None
    volatility: Optional[float] = None
    num_trades: Optional[int] = None
    # TRADING_METRICS original names
    sr: Optional[float] = None
    sor: Optional[float] = None
    cr: Optional[float] = None
    arr: Optional[float] = None
    mdd: Optional[float] = None
    vol: Optional[float] = None
    dd: Optional[float] = None
    # Per-symbol metrics (complete data for each symbol)
    # Each symbol contains: metrics + yearly_metrics + daily_values (per year)
    symbol_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    # Yearly aggregation statistics (across all symbols and years)
    yearly_metrics: Optional[List[Dict[str, Any]]] = None
    metrics_mean: Optional[Dict[str, float]] = None
    metrics_std: Optional[Dict[str, float]] = None
    metrics_median: Optional[Dict[str, float]] = None
    metrics_max: Optional[Dict[str, float]] = None
    metrics_min: Optional[Dict[str, float]] = None
    # Timing info
    execution_time_ms: Optional[float] = None
    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None  # "syntax" or "runtime"
    # Code and query
    strategy_code: Optional[str] = None
    factor_codes: Optional[List[str]] = None
    query_text: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Check if this sample passed (valid syntax and backtest)."""
        return self.syntax_valid and self.backtest_valid

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "model": self.model,
            "sample_id": self.sample_id,
            "syntax_valid": self.syntax_valid,
            "backtest_valid": self.backtest_valid,
            # Aggregated metrics
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "volatility": self.volatility,
            "num_trades": self.num_trades,
            # TRADING_METRICS original names
            "sr": self.sr,
            "sor": self.sor,
            "cr": self.cr,
            "arr": self.arr,
            "mdd": self.mdd,
            "vol": self.vol,
            "dd": self.dd,
            # Per-symbol metrics
            "symbol_metrics": self.symbol_metrics,
            # Yearly aggregation
            "yearly_metrics": self.yearly_metrics,
            "metrics_mean": self.metrics_mean,
            "metrics_std": self.metrics_std,
            "metrics_median": self.metrics_median,
            "metrics_max": self.metrics_max,
            "metrics_min": self.metrics_min,
            # Timing
            "execution_time_ms": self.execution_time_ms,
            # Error info
            "error": self.error,
            "error_type": self.error_type,
            # Code and query
            "strategy_code": self.strategy_code,
            "factor_codes": self.factor_codes,
            "query_text": self.query_text,
            "passed": self.passed,
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest runner."""

    data_path: str = "datasets/market"
    data_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "asset_name": "market",
            "data_type": "price",
            "level": "1day",
        },
        {
            "asset_name": "market",
            "data_type": "feature",
            "level": "1day",
        },
    ])
    start_ts: str = "2024-01-01 00:00:00"
    end_ts: str = "2024-12-31 00:00:00"
    history_ts: int = 120
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])


def _run_single_backtest_worker(args: Tuple[Dict[str, Any], int, "BacktestConfig"]) -> BacktestResult:
    """Worker function for multiprocessing - must be at module level."""
    sample, idx, config = args

    # Create a new runner instance in this process
    runner = BacktestRunner(config)
    return runner._run_single_sample(sample, idx, use_cache=False)


class BacktestRunner:
    """Run backtests on generated strategy code."""

    _trading_metrics_warning_shown = False  # Class-level flag for warning

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._cache: Dict[str, BacktestResult] = {}
        self._cache_file: Optional[Path] = None
        # 性能优化：批量保存缓存
        self._cache_dirty_count = 0
        self._cache_save_interval = 10  # 每10个样本保存一次
        # 性能优化：数据集缓存
        self._dataset_cache: Dict[str, Any] = {}
        # 线程安全：缓存锁
        self._cache_lock = threading.Lock()
        self._dataset_lock = threading.Lock()

    def set_cache_file(self, cache_file: Path):
        """Set cache file path and load existing cache."""
        # 改用 .jsonl 格式
        self._cache_file = cache_file.with_suffix('.jsonl')
        # 记录已保存的 key，用于增量写入
        self._saved_keys: set = set()
        if self._cache_file.exists():
            self._load_cache()

    def _load_cache(self):
        """Load cache from JSONL file."""
        if self._cache_file and self._cache_file.exists():
            try:
                valid_fields = {
                    'query_id', 'model', 'sample_id',
                    'syntax_valid', 'backtest_valid',
                    'sharpe', 'total_return', 'annual_return', 'max_drawdown',
                    'win_rate', 'sortino', 'calmar', 'volatility', 'num_trades',
                    'sr', 'sor', 'cr', 'arr', 'mdd', 'vol', 'dd',
                    'symbol_metrics', 'yearly_metrics',
                    'metrics_mean', 'metrics_std', 'metrics_median', 'metrics_max', 'metrics_min',
                    'execution_time_ms',
                    'error', 'error_type',
                    'strategy_code', 'factor_codes', 'query_text',
                }
                with open(self._cache_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            key = record.pop("_cache_key", None)
                            if not key:
                                continue
                            record.pop("passed", None)
                            filtered = {k: v for k, v in record.items() if k in valid_fields}
                            self._cache[key] = BacktestResult(**filtered)
                            self._saved_keys.add(key)
                        except (json.JSONDecodeError, TypeError):
                            continue
                logger.info(f"Loaded {len(self._cache)} cached backtest results")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """增量追加新结果到 JSONL 文件。"""
        with self._cache_lock:
            if self._cache_file:
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                new_keys = set(self._cache.keys()) - self._saved_keys
                if not new_keys:
                    return
                try:
                    with open(self._cache_file, "a") as f:
                        for key in new_keys:
                            record = self._cache[key].to_dict()
                            record["_cache_key"] = key
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            self._saved_keys.add(key)
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")
                self._cache_dirty_count = 0

    def _maybe_save_cache(self):
        """Save cache if dirty count exceeds interval."""
        should_save = False
        with self._cache_lock:
            self._cache_dirty_count += 1
            if self._cache_dirty_count >= self._cache_save_interval:
                should_save = True
        if should_save:
            self._save_cache()

    def _get_dataset(self, symbol: str, start_ts: str, end_ts: str):
        """Get or create cached dataset for a symbol and time range (thread-safe)."""
        from src.dataset.single_asset_dataset import SingleAssetDataset

        cache_key = f"{symbol}_{start_ts}_{end_ts}"
        with self._dataset_lock:
            if cache_key not in self._dataset_cache:
                self._dataset_cache[cache_key] = SingleAssetDataset(
                    symbol=symbol,
                    data_path=self.config.data_path,
                    enabled_data_configs=self.config.data_configs,
                    history_timestamps=self.config.history_ts,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    level="1day",
                )
            return self._dataset_cache[cache_key]

    def _get_cache_key(self, query_id: str, model: str, sample_id: int) -> str:
        """Generate cache key for a sample."""
        model_safe = model.replace("/", "_")
        return f"{model_safe}_{query_id}_{sample_id}"

    # Whitelist of allowed modules for LLM-generated code
    ALLOWED_MODULES = {
        "__future__",  # Allow annotations import
        "pandas", "numpy", "json", "math", "datetime", "typing",
        "collections", "itertools", "functools", "operator", "re",
        "pydantic", "src", "src.strategy", "src.strategy.types",
        "src.factor", "src.factor.types", "src.operator",
    }

    def _create_restricted_import(self):
        """Create a restricted __import__ function that only allows whitelisted modules."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Check if the module or its parent is in the whitelist
            module_root = name.split('.')[0]
            if module_root not in self.ALLOWED_MODULES and name not in self.ALLOWED_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed. Only strategy-related imports are permitted.")
            return original_import(name, globals, locals, fromlist, level)

        return restricted_import

    def validate_syntax(
        self,
        strategy_code: str,
        factor_codes: List[str],
    ) -> Tuple[bool, Optional[Any], List[Any], Optional[str]]:
        """
        Validate that code can be executed and classes instantiated.

        Returns:
            Tuple of (success, strategy_instance, factor_instances, error_message)
        """
        try:
            # Create restricted builtins with limited import
            import builtins
            restricted_builtins = {k: getattr(builtins, k) for k in dir(builtins)}
            restricted_builtins['__import__'] = self._create_restricted_import()

            # Create execution environment
            exec_globals = {
                "__builtins__": restricted_builtins,
                "pd": __import__("pandas"),
                "np": __import__("numpy"),
                "json": __import__("json"),
                "Field": __import__("pydantic").Field,
                "List": List,
                "Dict": Dict,
                "Any": Any,
                "Optional": Optional,
            }

            # Import strategy and factor types
            from src.strategy.types import Strategy
            from src.factor.types import Factor

            exec_globals["Strategy"] = Strategy
            exec_globals["Factor"] = Factor

            # Import operators (excluding builtins)
            try:
                from src import operator as ops

                builtin_names = {"min", "max", "abs", "sum", "any", "all"}
                for name in dir(ops):
                    if not name.startswith("_") and name not in builtin_names:
                        exec_globals[name] = getattr(ops, name)
            except Exception:
                pass

            # Execute factor codes first
            factor_classes = []
            for code in factor_codes:
                try:
                    exec(code, exec_globals)
                    # Find Factor subclasses
                    for name, obj in list(exec_globals.items()):
                        if isinstance(obj, type) and name != "Factor":
                            try:
                                if issubclass(obj, Factor) and obj is not Factor:
                                    if obj not in factor_classes:
                                        factor_classes.append(obj)
                            except TypeError:
                                pass
                except Exception as e:
                    logger.debug(f"Factor code execution failed: {e}")

            # Execute strategy code
            exec(strategy_code, exec_globals)

            # Find Strategy subclass
            strategy_class = None
            for name, obj in exec_globals.items():
                if isinstance(obj, type) and name != "Strategy":
                    try:
                        if issubclass(obj, Strategy) and obj is not Strategy:
                            strategy_class = obj
                            break
                    except TypeError:
                        pass

            if strategy_class is None:
                return False, None, [], "No Strategy subclass found"

            # Try to instantiate - 先调用 model_rebuild() 解析前向引用
            # 这是为了支持使用 `from __future__ import annotations` 的代码
            # 动态构建命名空间，自动包含所有 typing 类型，避免手动维护
            import typing
            import datetime as dt_module
            import pandas as pd_module
            import numpy as np_module

            # 自动导入 typing 模块的所有公开类型
            rebuild_namespace = {
                name: getattr(typing, name)
                for name in dir(typing)
                if not name.startswith('_')
            }
            # 添加常用模块和类型
            rebuild_namespace.update({
                "datetime": dt_module.datetime,
                "date": dt_module.date,
                "timedelta": dt_module.timedelta,
                "pd": pd_module,
                "np": np_module,
                "DataFrame": pd_module.DataFrame,
                "Series": pd_module.Series,
                "ndarray": np_module.ndarray,
            })

            try:
                strategy_class.model_rebuild(_types_namespace=rebuild_namespace)
            except Exception:
                pass  # 如果不需要 rebuild 则忽略

            for fc in factor_classes:
                try:
                    fc.model_rebuild(_types_namespace=rebuild_namespace)
                except Exception:
                    pass

            strategy_instance = strategy_class()
            factor_instances = [fc() for fc in factor_classes]

            return True, strategy_instance, factor_instances, None

        except SyntaxError as e:
            return False, None, [], f"Syntax error: {e}"
        except Exception as e:
            return False, None, [], f"Execution error: {str(e)[:200]}"

    def run_single_backtest(
        self,
        strategy,
        factor_instances: List[Any],
        symbol: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        data_start_ts: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Run backtest on a single symbol for a specific time range.

        Args:
            strategy: Strategy instance
            factor_instances: List of factor instances
            symbol: Symbol to backtest
            start_ts: Trading start timestamp (defaults to config.start_ts)
            end_ts: Trading end timestamp (defaults to config.end_ts)
            data_start_ts: Data loading start timestamp (defaults to start_ts - history_ts days)

        Returns:
            Tuple of (metrics_dict, error_message)
        """
        # Use provided time range or fall back to config
        trading_start = start_ts or self.config.start_ts
        trading_end = end_ts or self.config.end_ts

        # 计算数据加载起始时间
        if data_start_ts:
            data_start = data_start_ts
        else:
            # 默认向前扩展 history_ts 天
            trading_start_dt = datetime.strptime(trading_start, "%Y-%m-%d %H:%M:%S")
            data_start_dt = trading_start_dt - timedelta(days=self.config.history_ts)
            data_start = data_start_dt.strftime("%Y-%m-%d %H:%M:%S")

        try:
            from src.environment.trading import TradingEnvironment

            # 使用扩展的数据范围加载数据集
            dataset = self._get_dataset(symbol, data_start, trading_end)

            env = TradingEnvironment(
                dataset=dataset,
                history_timestamps=self.config.history_ts,
                step_timestamps=1,
                start_timestamp=trading_start,
                end_timestamp=trading_end,
            )

            state, _ = env.reset()
            values = [1.0]
            timestamps = []  # 记录时间戳
            signals = []
            prev_signal = 0

            while True:
                # state["data"] 已经是副本（来自 trading.py 的 _get_dataitem），无需再次复制
                df = state["data"]
                # 记录当前时间戳
                timestamps.append(str(df.index[-1]))

                # Calculate factors
                for factor in factor_instances:
                    try:
                        factor_out = factor(df)
                        # 如果返回的是协程，则运行它
                        if inspect.iscoroutine(factor_out):
                            factor_out = asyncio.run(factor_out)
                        if hasattr(factor_out, "columns"):
                            for col in factor_out.columns:
                                df[col] = factor_out[col]
                    except Exception:
                        pass

                # Call strategy
                result = strategy(df)
                # 如果返回的是协程，则运行它
                if inspect.iscoroutine(result):
                    result = asyncio.run(result)

                if isinstance(result, dict):
                    signal = result.get("signal", 0)
                    position_ratio = result.get("position", 1.0)
                else:
                    signal = int(result) if result else 0
                    position_ratio = 1.0

                signals.append(signal)
                state, reward, done, truncated, _ = env.step(signal, position_ratio)
                values.append(values[-1] * (1 + reward))
                prev_signal = signal

                if done or truncated:
                    break

            values = np.array(values)
            returns = np.diff(values) / values[:-1]

            # Calculate all metrics
            metrics = self._calculate_metrics(values, returns, signals)
            # Save daily values and timestamps for plotting
            metrics['daily_values'] = values.tolist()
            metrics['daily_timestamps'] = timestamps
            return metrics, None

        except Exception as e:
            return None, str(e)[:200]

    def _calculate_metrics(
        self,
        values: np.ndarray,
        returns: np.ndarray,
        signals: List[int]
    ) -> Dict[str, Any]:
        """Calculate all backtest metrics using TRADING_METRICS."""
        metrics = {}

        # Try to use TRADING_METRICS from src.metric
        try:
            from src.metric import TRADING_METRICS

            for metric_cls in TRADING_METRICS:
                metric_name = metric_cls.__name__
                try:
                    if metric_name in ['SR', 'SOR']:
                        metric_func = metric_cls(
                            level="1day",
                            symbol_info={"exchange": "crypto"},
                            risk_free_rate=0.0
                        )
                    else:
                        metric_func = metric_cls(
                            level="1day",
                            symbol_info={"exchange": "crypto"}
                        )

                    value = metric_func(returns)
                    if not np.isnan(value) and not np.isinf(value):
                        metrics[metric_name.lower()] = float(value)
                    else:
                        metrics[metric_name.lower()] = None
                except Exception as e:
                    logger.debug(f"Failed to calculate {metric_name}: {e}")
                    metrics[metric_name.lower()] = None

        except ImportError:
            # Fallback to manual calculation if TRADING_METRICS not available
            if not BacktestRunner._trading_metrics_warning_shown:
                logger.warning("TRADING_METRICS not available (missing pandas_market_calendars), using manual calculation")
                BacktestRunner._trading_metrics_warning_shown = True

        # Manual calculations for metrics not in TRADING_METRICS or as fallback
        std = np.std(returns) if len(returns) > 0 else 0
        mean_return = np.mean(returns) if len(returns) > 0 else 0

        # Sharpe ratio (annualized) - fallback if SR not calculated
        if 'sr' not in metrics or metrics['sr'] is None:
            sharpe = float(mean_return / std * np.sqrt(252)) if std > 0 else 0.0
            metrics['sr'] = sharpe

        # Total return (not in TRADING_METRICS)
        total_return = float((values[-1] / values[0]) - 1) if len(values) > 0 else 0.0
        metrics['total_return'] = total_return

        # Annual return - fallback if ARR not calculated
        num_days = len(returns)
        metrics['num_days'] = num_days  # 保存交易天数用于聚合计算
        if 'arr' not in metrics or metrics['arr'] is None:
            annual_return = float((1 + total_return) ** (252 / num_days) - 1) if num_days > 0 else 0.0
            metrics['arr'] = annual_return

        # Volatility - fallback if VOL not calculated
        if 'vol' not in metrics or metrics['vol'] is None:
            volatility = float(std * np.sqrt(252))
            metrics['vol'] = volatility

        # Max drawdown - fallback if MDD not calculated
        if 'mdd' not in metrics or metrics['mdd'] is None:
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            max_drawdown = float(np.min(drawdown))
            metrics['mdd'] = max_drawdown

        # Sortino ratio - fallback if SOR not calculated
        if 'sor' not in metrics or metrics['sor'] is None:
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
            sortino = float(mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
            metrics['sor'] = sortino

        # Calmar ratio - fallback if CR not calculated
        if 'cr' not in metrics or metrics['cr'] is None:
            mdd_val = metrics.get('mdd', 0)
            arr_val = metrics.get('arr', 0)
            calmar = float(arr_val / abs(mdd_val)) if mdd_val != 0 else 0.0
            metrics['cr'] = calmar

        # Win rate (not in TRADING_METRICS)
        winning_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = float(winning_days / total_days) if total_days > 0 else 0.0
        metrics['win_rate'] = win_rate

        # Number of trades (signal changes, not in TRADING_METRICS)
        signals_arr = np.array(signals)
        num_trades = int(np.sum(np.diff(signals_arr) != 0)) if len(signals_arr) > 1 else 0
        metrics['num_trades'] = num_trades

        # Create backward-compatible field names
        metrics['sharpe'] = metrics.get('sr')
        metrics['sortino'] = metrics.get('sor')
        metrics['calmar'] = metrics.get('cr')
        metrics['annual_return'] = metrics.get('arr')
        metrics['max_drawdown'] = metrics.get('mdd')
        metrics['volatility'] = metrics.get('vol')

        return metrics

    def _split_years(self) -> List[Tuple[str, str, str, int]]:
        """Split time range into yearly periods with extended data range.

        Returns:
            List of (data_start_ts, trading_start_ts, trading_end_ts, year) tuples
            - data_start_ts: 数据加载起始时间（向前扩展 history_ts 天）
            - trading_start_ts: 实际交易起始时间
            - trading_end_ts: 实际交易结束时间
            - year: 年份
        """
        start = datetime.strptime(self.config.start_ts, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(self.config.end_ts, "%Y-%m-%d %H:%M:%S")

        periods = []
        current_year = start.year
        while current_year <= end.year:
            trading_start = f"{current_year}-01-01 00:00:00"
            trading_end = f"{current_year}-12-31 23:59:59"

            # Clamp to actual range
            if current_year == start.year:
                trading_start = self.config.start_ts
            if current_year == end.year:
                trading_end = self.config.end_ts

            # 计算数据加载起始时间（向前扩展 history_ts 天）
            trading_start_dt = datetime.strptime(trading_start, "%Y-%m-%d %H:%M:%S")
            data_start_dt = trading_start_dt - timedelta(days=self.config.history_ts)
            data_start = data_start_dt.strftime("%Y-%m-%d %H:%M:%S")

            periods.append((data_start, trading_start, trading_end, current_year))
            current_year += 1

        return periods

    def _aggregate_yearly_metrics(self, yearly_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate yearly metrics into statistics.

        Args:
            yearly_results: List of metrics dicts, each with a 'year' key

        Returns:
            Dict with keys 'mean', 'std', 'median', 'max', 'min', each containing metric aggregations
        """
        if not yearly_results:
            return {'mean': {}, 'std': {}, 'median': {}, 'max': {}, 'min': {}}

        # 可以直接平均的指标
        avg_keys = ['sharpe', 'sr', 'sortino', 'sor', 'win_rate', 'volatility', 'vol', 'max_drawdown', 'mdd', 'dd']
        # 需要重新计算的指标（依赖其他指标）
        derived_keys = ['annual_return', 'arr', 'calmar', 'cr']

        aggregated = {'mean': {}, 'std': {}, 'median': {}, 'max': {}, 'min': {}}

        # 1. 可以直接平均的指标
        for key in avg_keys:
            values = [r.get(key) for r in yearly_results if r.get(key) is not None]
            if values:
                aggregated['mean'][key] = float(np.mean(values))
                aggregated['std'][key] = float(np.std(values))
                aggregated['median'][key] = float(np.median(values))
                aggregated['max'][key] = float(np.max(values))
                aggregated['min'][key] = float(np.min(values))

        # 2. 总收益率：用复合收益计算
        total_returns = [r.get('total_return') for r in yearly_results if r.get('total_return') is not None]
        if total_returns:
            compound_return = np.prod([1 + r for r in total_returns]) - 1
            aggregated['mean']['total_return'] = float(compound_return)
            # std/median/max/min 仍用原始值
            aggregated['std']['total_return'] = float(np.std(total_returns))
            aggregated['median']['total_return'] = float(np.median(total_returns))
            aggregated['max']['total_return'] = float(np.max(total_returns))
            aggregated['min']['total_return'] = float(np.min(total_returns))

        # 3. 年化收益率：各年收益率的算术平均
        arr_values = [r.get('arr') or r.get('annual_return') for r in yearly_results
                      if (r.get('arr') or r.get('annual_return')) is not None]
        if arr_values:
            aggregated['mean']['arr'] = float(np.mean(arr_values))
            aggregated['mean']['annual_return'] = float(np.mean(arr_values))

        # 4. Calmar：基于正确的年化收益和最大回撤重新计算
        arr_val = aggregated['mean'].get('arr', 0)
        mdd_val = aggregated['mean'].get('mdd', 0)
        if mdd_val != 0:
            calmar = arr_val / abs(mdd_val)
            aggregated['mean']['cr'] = float(calmar)
            aggregated['mean']['calmar'] = float(calmar)

        return aggregated

    def _aggregate_symbol_metrics(self, symbol_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all symbols.

        Args:
            symbol_data_list: List of symbol metrics dicts

        Returns:
            Dict with averaged metrics across all symbols
        """
        if not symbol_data_list:
            return {}

        aggregated = {}

        # 可以直接平均的指标
        avg_keys = ['sharpe', 'sr', 'sortino', 'sor', 'win_rate', 'volatility', 'vol']
        for key in avg_keys:
            values = [d.get(key) for d in symbol_data_list if d.get(key) is not None]
            if values:
                aggregated[key] = float(np.mean(values))

        # 总收益率：多个币种取平均（假设等权重）
        total_returns = [d.get('total_return') for d in symbol_data_list if d.get('total_return') is not None]
        if total_returns:
            aggregated['total_return'] = float(np.mean(total_returns))

        # 最大回撤：算术平均
        mdd_values = [d.get('mdd') or d.get('max_drawdown') for d in symbol_data_list
                      if (d.get('mdd') or d.get('max_drawdown')) is not None]
        if mdd_values:
            aggregated['mdd'] = float(np.mean(mdd_values))
            aggregated['max_drawdown'] = aggregated['mdd']

        # 年化收益率：各年收益率的算术平均
        arr_values = [d.get('arr') or d.get('annual_return') for d in symbol_data_list
                      if (d.get('arr') or d.get('annual_return')) is not None]
        if arr_values:
            aggregated['arr'] = float(np.mean(arr_values))
            aggregated['annual_return'] = float(np.mean(arr_values))

        # Calmar：重新计算
        arr_val = aggregated.get('arr', 0)
        mdd_val = aggregated.get('mdd', 0)
        if mdd_val != 0:
            aggregated['cr'] = float(arr_val / abs(mdd_val))
            aggregated['calmar'] = aggregated['cr']

        # num_trades
        trades = [d.get('num_trades') for d in symbol_data_list if d.get('num_trades') is not None]
        if trades:
            aggregated['num_trades'] = int(np.sum(trades))

        return aggregated

    def run_backtest(
        self,
        query_id: str,
        model: str,
        sample_id: int,
        strategy_code: str,
        factor_codes: List[str],
        query_text: str = "",
        use_cache: bool = True,
    ) -> BacktestResult:
        """
        Run full backtest validation on a code sample.

        This method:
        1. Validates syntax
        2. Runs backtest for each symbol, split by year
        3. Aggregates metrics per symbol (with yearly statistics)
        4. Aggregates metrics across all symbols
        5. Records execution time

        Args:
            query_id: Query identifier
            model: Model name
            sample_id: Sample index
            strategy_code: Strategy class code
            factor_codes: List of factor class codes
            query_text: Original query text
            use_cache: Whether to use cached results

        Returns:
            BacktestResult object with complete metrics
        """
        start_time = time.perf_counter()

        cache_key = self._get_cache_key(query_id, model, sample_id)

        # Check cache (thread-safe)
        with self._cache_lock:
            if use_cache and cache_key in self._cache:
                cached = self._cache[cache_key]
                # Skip cache if it's a runtime error (likely missing factor that may now exist)
                # Runtime errors like "'max_14'" indicate missing factors that factor_validator
                # may have computed since the cache was created
                if cached.error_type == "runtime" and cached.error:
                    # Don't use cached runtime errors - retry the backtest
                    pass
                else:
                    # Update code/query if not present in cached result
                    if cached.strategy_code is None:
                        cached.strategy_code = strategy_code
                        cached.factor_codes = factor_codes
                        cached.query_text = query_text
                    return cached

        # Validate syntax first
        syntax_valid, strategy, factors, syntax_error = self.validate_syntax(
            strategy_code, factor_codes
        )

        if not syntax_valid:
            end_time = time.perf_counter()
            result = BacktestResult(
                query_id=query_id,
                model=model,
                sample_id=sample_id,
                syntax_valid=False,
                backtest_valid=False,
                error=syntax_error,
                error_type="syntax",
                strategy_code=strategy_code,
                factor_codes=factor_codes,
                query_text=query_text,
                execution_time_ms=(end_time - start_time) * 1000,
            )
            with self._cache_lock:
                self._cache[cache_key] = result
            self._maybe_save_cache()
            return result

        # Get yearly periods
        yearly_periods = self._split_years()

        # Run backtest for each symbol with yearly split
        symbol_metrics: Dict[str, Dict[str, Any]] = {}
        all_yearly_metrics: List[Dict[str, Any]] = []
        errors = []

        for symbol in self.config.symbols:
            yearly_results_for_symbol: List[Dict[str, Any]] = []

            for data_start, trading_start, trading_end, year in yearly_periods:
                try:
                    metrics, error = self.run_single_backtest(
                        strategy, factors, symbol,
                        start_ts=trading_start,
                        end_ts=trading_end,
                        data_start_ts=data_start,
                    )
                    if metrics is not None:
                        metrics['year'] = year
                        metrics['symbol'] = symbol
                        yearly_results_for_symbol.append(metrics)
                        all_yearly_metrics.append(metrics)
                    elif error:
                        errors.append(f"{symbol}/{year}: {error}")
                except Exception as e:
                    errors.append(f"{symbol}/{year}: {str(e)[:100]}")

            # Aggregate yearly metrics for this symbol
            if yearly_results_for_symbol:
                yearly_agg = self._aggregate_yearly_metrics(yearly_results_for_symbol)

                # Use the mean of yearly metrics as the symbol's main metrics
                symbol_main_metrics = yearly_agg['mean'].copy()
                symbol_main_metrics['num_trades'] = int(np.mean([
                    r.get('num_trades', 0) for r in yearly_results_for_symbol
                    if r.get('num_trades') is not None
                ])) if yearly_results_for_symbol else 0

                symbol_metrics[symbol] = {
                    **symbol_main_metrics,
                    'yearly_metrics': yearly_results_for_symbol,
                    'metrics_mean': yearly_agg['mean'],
                    'metrics_std': yearly_agg['std'],
                    'metrics_median': yearly_agg['median'],
                    'metrics_max': yearly_agg['max'],
                    'metrics_min': yearly_agg['min'],
                }

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        if symbol_metrics:
            # Aggregate across all symbols
            all_symbol_data = [
                {k: v for k, v in sm.items() if k not in ['yearly_metrics', 'metrics_mean', 'metrics_std', 'metrics_median', 'metrics_max', 'metrics_min']}
                for sm in symbol_metrics.values()
            ]
            final_aggregated = self._aggregate_symbol_metrics(all_symbol_data)

            # Aggregate yearly statistics across all symbols
            overall_yearly_agg = self._aggregate_yearly_metrics(all_yearly_metrics)

            result = BacktestResult(
                query_id=query_id,
                model=model,
                sample_id=sample_id,
                syntax_valid=True,
                backtest_valid=True,
                # Aggregated metrics (average across all symbols)
                sharpe=final_aggregated.get("sharpe"),
                total_return=final_aggregated.get("total_return"),
                annual_return=final_aggregated.get("annual_return"),
                max_drawdown=final_aggregated.get("max_drawdown"),
                win_rate=final_aggregated.get("win_rate"),
                sortino=final_aggregated.get("sortino"),
                calmar=final_aggregated.get("calmar"),
                volatility=final_aggregated.get("volatility"),
                num_trades=final_aggregated.get("num_trades"),
                # TRADING_METRICS original names
                sr=final_aggregated.get("sr"),
                sor=final_aggregated.get("sor"),
                cr=final_aggregated.get("cr"),
                arr=final_aggregated.get("arr"),
                mdd=final_aggregated.get("mdd"),
                vol=final_aggregated.get("vol"),
                dd=final_aggregated.get("dd"),
                # Per-symbol metrics
                symbol_metrics=symbol_metrics,
                # Yearly aggregation (across all symbols and years)
                yearly_metrics=all_yearly_metrics,
                metrics_mean=overall_yearly_agg['mean'],
                metrics_std=overall_yearly_agg['std'],
                metrics_median=overall_yearly_agg['median'],
                metrics_max=overall_yearly_agg['max'],
                metrics_min=overall_yearly_agg['min'],
                # Timing
                execution_time_ms=execution_time_ms,
                # Code and query
                strategy_code=strategy_code,
                factor_codes=factor_codes,
                query_text=query_text,
            )
        else:
            result = BacktestResult(
                query_id=query_id,
                model=model,
                sample_id=sample_id,
                syntax_valid=True,
                backtest_valid=False,
                error=errors[0] if errors else "No valid backtest results",
                error_type="runtime",
                strategy_code=strategy_code,
                factor_codes=factor_codes,
                query_text=query_text,
                execution_time_ms=execution_time_ms,
            )

        with self._cache_lock:
            self._cache[cache_key] = result
        self._save_cache()
        return result

    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple sources (legacy method for compatibility)."""
        if not metrics_list:
            return {}

        # Average all float metrics
        aggregated = {}
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

        for key in all_keys:
            values = [m.get(key) for m in metrics_list if m.get(key) is not None]
            if values:
                if key == "num_trades":
                    aggregated[key] = int(np.mean(values))
                elif key in ['year', 'symbol']:
                    continue  # Skip non-numeric fields
                else:
                    try:
                        aggregated[key] = float(np.mean(values))
                    except (TypeError, ValueError):
                        continue
            else:
                aggregated[key] = None
        return aggregated

    def run_batch_backtest(
        self,
        samples: List[Dict[str, Any]],
        use_cache: bool = True,
        num_workers: int = 1,
        timeout: int = 10800,
    ) -> List[BacktestResult]:
        """
        Run backtests on a batch of samples.

        Args:
            samples: List of sample dicts with extracted code
            use_cache: Whether to use cached results
            num_workers: Number of parallel workers (1 = serial execution)
            timeout: Timeout in seconds for batch backtest (default: 10800 = 3 hours)

        Returns:
            List of BacktestResult objects
        """
        if num_workers <= 1:
            return self._run_batch_serial(samples, use_cache)
        else:
            return self._run_batch_parallel(samples, use_cache, num_workers, timeout)

    def _run_single_sample(
        self,
        sample: Dict[str, Any],
        index: int,
        use_cache: bool,
    ) -> BacktestResult:
        """Run backtest on a single sample (used by both serial and parallel modes)."""
        query_id = sample.get("query_id", f"unknown_{index}")
        model = sample.get("model", "unknown")
        sample_id = sample.get("sample_id", 0)
        query_text = sample.get("query_text", "")
        strategy_code = sample.get("strategy_code", "")
        factor_codes = sample.get("factor_codes", [])

        if not strategy_code:
            return BacktestResult(
                query_id=query_id,
                model=model,
                sample_id=sample_id,
                syntax_valid=False,
                backtest_valid=False,
                error="No strategy code provided",
                error_type="syntax",
                strategy_code=strategy_code,
                factor_codes=factor_codes,
                query_text=query_text,
            )
        else:
            return self.run_backtest(
                query_id=query_id,
                model=model,
                sample_id=sample_id,
                strategy_code=strategy_code,
                factor_codes=factor_codes,
                query_text=query_text,
                use_cache=use_cache,
            )

    def _run_batch_serial(
        self,
        samples: List[Dict[str, Any]],
        use_cache: bool,
    ) -> List[BacktestResult]:
        """Run backtests serially (single-threaded)."""
        results = []
        execution_times = []
        last_execution_time = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[cyan]Last: {task.fields[last_time]}ms"),
            TextColumn("•"),
            TextColumn("[green]Avg: {task.fields[avg_time]}ms"),
        ) as progress:
            task_id = progress.add_task(
                "Running backtests",
                total=len(samples),
                last_time="--",
                avg_time="--",
            )

            for i, sample in enumerate(samples):
                model = sample.get("model", "unknown")
                query_id = sample.get("query_id", f"unknown_{i}")

                model_short = model.split("/")[-1] if "/" in model else model
                progress.update(
                    task_id,
                    description=f"[{model_short}] {query_id}"
                )

                result = self._run_single_sample(sample, i, use_cache)

                if result.execution_time_ms is not None:
                    last_execution_time = result.execution_time_ms
                    execution_times.append(last_execution_time)

                avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
                last_time_str = f"{last_execution_time:.0f}" if last_execution_time > 0 else "--"
                avg_time_str = f"{avg_time:.0f}" if avg_time > 0 else "--"

                results.append(result)
                progress.update(
                    task_id,
                    advance=1,
                    last_time=last_time_str,
                    avg_time=avg_time_str,
                )

        self._print_summary(results, execution_times)
        return results

    def _run_batch_parallel(
        self,
        samples: List[Dict[str, Any]],
        use_cache: bool,
        num_workers: int,
        timeout: int = 10800,
    ) -> List[BacktestResult]:
        """Run backtests in parallel using ProcessPoolExecutor."""
        # 限制最大 worker 数量，避免 "Too many open files" 错误
        max_allowed_workers = 256
        if num_workers > max_allowed_workers:
            logger.warning(f"Reducing workers from {num_workers} to {max_allowed_workers} to avoid system limits")
            num_workers = max_allowed_workers

        results: List[Optional[BacktestResult]] = [None] * len(samples)
        execution_times = []
        start_time = time.perf_counter()

        # Check cache first in main process
        samples_to_run = []  # (original_idx, sample, config)
        cached_count = 0
        for idx, sample in enumerate(samples):
            if use_cache:
                cache_key = self._get_cache_key(
                    sample.get("query_id", f"unknown_{idx}"),
                    sample.get("model", "unknown"),
                    sample.get("sample_id", 0),
                )
                if cache_key in self._cache:
                    cached = self._cache[cache_key]
                    # Skip runtime errors (may be fixed now)
                    if not (cached.error_type == "runtime" and cached.error):
                        results[idx] = cached
                        cached_count += 1
                        continue
            samples_to_run.append((idx, sample, self.config))

        if cached_count > 0:
            logger.info(f"Using {cached_count} cached results, running {len(samples_to_run)} new backtests")

        # If all samples are cached, skip parallel execution
        if not samples_to_run:
            final_results = [r for r in results if r is not None]
            self._print_summary(final_results, execution_times, num_workers)
            return final_results

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[cyan]Pending: {task.fields[pending]}"),
            TextColumn("•"),
            TextColumn("[green]Speed: {task.fields[speed]}"),
            TextColumn("•"),
            TextColumn("[yellow]Pass: {task.fields[pass_rate]}"),
        )
        progress.start()
        task_id = progress.add_task(
            f"Running backtests ({num_workers} workers)",
            total=len(samples),
            completed=cached_count,
            pending=len(samples_to_run),
            speed="--",
            pass_rate="--",
        )

        total_tasks = len(samples)
        last_save_pct = 0

        # 记录卡住的任务索引，用于串行重试
        stuck_indices = []

        executor = ProcessPoolExecutor(max_workers=num_workers)
        try:
            future_to_idx = {
                executor.submit(_run_single_backtest_worker, (sample, orig_idx, config)): orig_idx
                for orig_idx, sample, config in samples_to_run
            }

            # Use as_completed with a global timeout to avoid hanging
            pending_futures = set(future_to_idx.keys())
            no_progress_count = 0
            last_pending_count = len(pending_futures)

            while pending_futures:
                # Wait for any future to complete, with timeout
                done_futures = set()
                for future in list(pending_futures):
                    if future.done():
                        done_futures.add(future)

                if not done_futures:
                    # No futures completed yet, wait a bit
                    time.sleep(0.1)
                    # Check for overall timeout
                    elapsed = time.perf_counter() - start_time
                    if elapsed > timeout:
                        timeout_mins = timeout // 60
                        logger.warning(f"Timeout: {len(pending_futures)} tasks still pending after {timeout_mins} minutes")
                        break

                    # Check for stuck workers (120 seconds)
                    pending_count = len(pending_futures)
                    stuck_threshold = 1200  # 120 seconds

                    if pending_count == last_pending_count:
                        no_progress_count += 1
                        if no_progress_count >= stuck_threshold:
                            # 收集卡住的任务索引，稍后串行重试
                            stuck_indices = []
                            stuck_tasks = []
                            for future in pending_futures:
                                idx = future_to_idx[future]
                                stuck_indices.append(idx)
                                sample = samples[idx]
                                stuck_tasks.append(f"{sample.get('model', '?')}_{sample.get('query_id', '?')}_{sample.get('sample_id', '?')}")
                            logger.warning(f"No progress for 120s, {pending_count} tasks stuck, will retry serially: {stuck_tasks}")
                            break
                    else:
                        no_progress_count = 0
                        last_pending_count = len(pending_futures)
                    continue

                for future in done_futures:
                    pending_futures.discard(future)
                    idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=1)
                        results[idx] = result
                        if result.execution_time_ms is not None:
                            execution_times.append(result.execution_time_ms)
                        # Add to cache immediately
                        cache_key = self._get_cache_key(
                            result.query_id, result.model, result.sample_id
                        )
                        self._cache[cache_key] = result
                    except Exception as e:
                        sample = samples[idx]
                        results[idx] = BacktestResult(
                            query_id=sample.get("query_id", f"unknown_{idx}"),
                            model=sample.get("model", "unknown"),
                            sample_id=sample.get("sample_id", 0),
                            syntax_valid=False,
                            backtest_valid=False,
                            error=str(e)[:200],
                            error_type="runtime",
                        )

                elapsed = time.perf_counter() - start_time
                completed = sum(1 for r in results if r is not None)
                passed = sum(1 for r in results if r is not None and r.backtest_valid)
                # 速度只计算本次实际运行的任务（排除缓存）
                actually_completed = completed - cached_count
                speed_str = f"{actually_completed / elapsed:.1f}/s" if elapsed > 0 else "--"
                pass_rate_str = f"{passed}/{completed}" if completed > 0 else "--"
                progress.update(task_id, completed=completed, pending=len(pending_futures), speed=speed_str, pass_rate=pass_rate_str)

                # Save cache every 10% progress
                current_pct = int(completed * 100 / total_tasks) if total_tasks > 0 else 0
                if current_pct >= last_save_pct + 10:
                    self._save_cache()
                    last_save_pct = current_pct

                # 有任务完成，重置无进展计数器
                no_progress_count = 0
                last_pending_count = len(pending_futures)
        finally:
            progress.stop()
            print()  # 换行
            print("✓ Backtests complete, finalizing...", flush=True)
            print("  → Shutting down workers...", end=" ", flush=True)
            # wait=False 立即返回，不等待卡住的任务
            executor.shutdown(wait=False, cancel_futures=True)
            print("done", flush=True)

        # 串行重试卡住的任务
        if stuck_indices:
            print(f"  → Retrying {len(stuck_indices)} stuck tasks serially...", flush=True)
            for idx in stuck_indices:
                sample = samples[idx]
                result = self._run_single_sample(sample, idx, use_cache=False)
                results[idx] = result
                if result.execution_time_ms:
                    execution_times.append(result.execution_time_ms)
                # 更新缓存
                cache_key = self._get_cache_key(
                    result.query_id, result.model, result.sample_id
                )
                self._cache[cache_key] = result
                status = "✓" if result.backtest_valid else "✗"
                print(f"    {status} {sample.get('model', '?')}_{sample.get('query_id', '?')}_{sample.get('sample_id', '?')}", flush=True)
            self._save_cache()

        # Save results to cache in main process
        final_results = [r for r in results if r is not None]
        print(f"  → Saving {len(final_results)} results to cache...", end=" ", flush=True)
        for result in final_results:
            cache_key = self._get_cache_key(
                result.query_id, result.model, result.sample_id
            )
            self._cache[cache_key] = result
        self._save_cache()
        print("done", flush=True)
        print("  → Generating summary...", flush=True)

        self._print_summary(final_results, execution_times, num_workers)
        return final_results

    def _print_summary(
        self,
        results: List[BacktestResult],
        execution_times: List[float],
        num_workers: int = 1,
    ) -> None:
        """Print final summary and save cache."""
        if execution_times:
            total_time = sum(execution_times)
            avg_time = total_time / len(execution_times)
            if num_workers > 1:
                # 并行模式：显示实际吞吐量
                actual_time = total_time / num_workers  # 估算实际耗时
                throughput = len(results) / (actual_time / 1000) if actual_time > 0 else 0
                logger.info(
                    f"Backtest complete: {len(results)} samples, "
                    f"avg={avg_time:.0f}ms/task, ~{throughput:.1f} samples/s"
                )
            else:
                logger.info(
                    f"Backtest complete: {len(results)} samples, "
                    f"avg={avg_time:.0f}ms, total={total_time/1000:.1f}s"
                )

        # 确保最后的缓存被保存
        with self._cache_lock:
            if self._cache_dirty_count > 0:
                self._save_cache()
