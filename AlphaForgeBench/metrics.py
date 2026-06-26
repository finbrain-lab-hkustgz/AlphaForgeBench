"""Metrics calculation for Pass@k evaluation."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from AlphaForgeBench.backtest_runner import BacktestResult


@dataclass
class QueryMetrics:
    """Metrics for a single query across all samples."""

    query_id: str
    level: str
    num_samples: int = 0
    num_passed: int = 0
    num_syntax_valid: int = 0
    first_sample_passed: bool = False

    # Collected metrics from all passed samples
    sharpes: List[float] = field(default_factory=list)
    total_returns: List[float] = field(default_factory=list)
    annual_returns: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    win_rates: List[float] = field(default_factory=list)
    sortinos: List[float] = field(default_factory=list)
    calmars: List[float] = field(default_factory=list)
    volatilities: List[float] = field(default_factory=list)
    num_trades_list: List[int] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)

    @property
    def pass_at_k(self) -> float:
        """Pass@k: 1 if any sample passed, 0 otherwise."""
        return 1.0 if self.num_passed > 0 else 0.0

    @property
    def pass_at_1(self) -> float:
        """Pass@1: 1 if first sample passed, 0 otherwise."""
        return 1.0 if self.first_sample_passed else 0.0

    @property
    def syntax_pass_rate(self) -> float:
        """Proportion of samples with valid syntax."""
        return self.num_syntax_valid / self.num_samples if self.num_samples > 0 else 0.0

    @property
    def avg_sharpe(self) -> Optional[float]:
        """Average Sharpe ratio of passed samples."""
        return float(np.mean(self.sharpes)) if self.sharpes else None

    @property
    def avg_total_return(self) -> Optional[float]:
        """Average total return of passed samples."""
        return float(np.mean(self.total_returns)) if self.total_returns else None

    @property
    def avg_annual_return(self) -> Optional[float]:
        """Average annual return of passed samples."""
        return float(np.mean(self.annual_returns)) if self.annual_returns else None

    @property
    def avg_max_drawdown(self) -> Optional[float]:
        """Average max drawdown of passed samples."""
        return float(np.mean(self.max_drawdowns)) if self.max_drawdowns else None

    @property
    def avg_win_rate(self) -> Optional[float]:
        """Average win rate of passed samples."""
        return float(np.mean(self.win_rates)) if self.win_rates else None

    @property
    def avg_sortino(self) -> Optional[float]:
        """Average Sortino ratio of passed samples."""
        return float(np.mean(self.sortinos)) if self.sortinos else None

    @property
    def avg_calmar(self) -> Optional[float]:
        """Average Calmar ratio of passed samples."""
        return float(np.mean(self.calmars)) if self.calmars else None

    @property
    def avg_volatility(self) -> Optional[float]:
        """Average volatility of passed samples."""
        return float(np.mean(self.volatilities)) if self.volatilities else None

    @property
    def avg_num_trades(self) -> Optional[float]:
        """Average number of trades of passed samples."""
        return float(np.mean(self.num_trades_list)) if self.num_trades_list else None

    @property
    def avg_execution_time_ms(self) -> Optional[float]:
        """Average execution time in milliseconds."""
        return float(np.mean(self.execution_times)) if self.execution_times else None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "level": self.level,
            "num_samples": self.num_samples,
            "num_passed": self.num_passed,
            "num_syntax_valid": self.num_syntax_valid,
            "pass_at_k": self.pass_at_k,
            "pass_at_1": self.pass_at_1,
            "syntax_pass_rate": self.syntax_pass_rate,
            # Aggregated metrics (mean across samples)
            "avg_sharpe": self.avg_sharpe,
            "avg_total_return": self.avg_total_return,
            "avg_annual_return": self.avg_annual_return,
            "avg_max_drawdown": self.avg_max_drawdown,
            "avg_win_rate": self.avg_win_rate,
            "avg_sortino": self.avg_sortino,
            "avg_calmar": self.avg_calmar,
            "avg_volatility": self.avg_volatility,
            "avg_num_trades": self.avg_num_trades,
            "avg_execution_time_ms": self.avg_execution_time_ms,
        }


@dataclass
class LevelMetrics:
    """Aggregated metrics for a difficulty level."""

    level: str
    num_queries: int = 0
    pass_at_k_sum: float = 0.0
    pass_at_1_sum: float = 0.0
    syntax_pass_sum: float = 0.0
    sharpes: List[float] = field(default_factory=list)
    total_returns: List[float] = field(default_factory=list)
    annual_returns: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    win_rates: List[float] = field(default_factory=list)
    sortinos: List[float] = field(default_factory=list)
    calmars: List[float] = field(default_factory=list)
    volatilities: List[float] = field(default_factory=list)
    num_trades_list: List[int] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)

    @property
    def pass_at_k(self) -> float:
        """Average Pass@k across all queries in this level."""
        return self.pass_at_k_sum / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def pass_at_1(self) -> float:
        """Average Pass@1 across all queries in this level."""
        return self.pass_at_1_sum / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def syntax_pass_rate(self) -> float:
        """Average syntax pass rate across all queries."""
        return self.syntax_pass_sum / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def avg_sharpe(self) -> Optional[float]:
        return float(np.mean(self.sharpes)) if self.sharpes else None

    @property
    def avg_total_return(self) -> Optional[float]:
        return float(np.mean(self.total_returns)) if self.total_returns else None

    @property
    def avg_annual_return(self) -> Optional[float]:
        return float(np.mean(self.annual_returns)) if self.annual_returns else None

    @property
    def avg_max_drawdown(self) -> Optional[float]:
        return float(np.mean(self.max_drawdowns)) if self.max_drawdowns else None

    @property
    def avg_win_rate(self) -> Optional[float]:
        return float(np.mean(self.win_rates)) if self.win_rates else None

    @property
    def avg_sortino(self) -> Optional[float]:
        return float(np.mean(self.sortinos)) if self.sortinos else None

    @property
    def avg_calmar(self) -> Optional[float]:
        return float(np.mean(self.calmars)) if self.calmars else None

    @property
    def avg_volatility(self) -> Optional[float]:
        return float(np.mean(self.volatilities)) if self.volatilities else None

    @property
    def avg_num_trades(self) -> Optional[float]:
        return float(np.mean(self.num_trades_list)) if self.num_trades_list else None

    @property
    def avg_execution_time_ms(self) -> Optional[float]:
        return float(np.mean(self.execution_times)) if self.execution_times else None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "num_queries": self.num_queries,
            "pass_at_k": self.pass_at_k,
            "pass_at_1": self.pass_at_1,
            "syntax_pass_rate": self.syntax_pass_rate,
            "avg_sharpe": self.avg_sharpe,
            "avg_total_return": self.avg_total_return,
            "avg_annual_return": self.avg_annual_return,
            "avg_max_drawdown": self.avg_max_drawdown,
            "avg_win_rate": self.avg_win_rate,
            "avg_sortino": self.avg_sortino,
            "avg_calmar": self.avg_calmar,
            "avg_volatility": self.avg_volatility,
            "avg_num_trades": self.avg_num_trades,
            "avg_execution_time_ms": self.avg_execution_time_ms,
        }


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model."""

    model: str
    num_queries: int = 0
    pass_at_k_sum: float = 0.0
    pass_at_1_sum: float = 0.0
    syntax_pass_sum: float = 0.0
    sharpes: List[float] = field(default_factory=list)
    total_returns: List[float] = field(default_factory=list)
    annual_returns: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    win_rates: List[float] = field(default_factory=list)
    sortinos: List[float] = field(default_factory=list)
    calmars: List[float] = field(default_factory=list)
    volatilities: List[float] = field(default_factory=list)
    num_trades_list: List[int] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)
    by_level: Dict[str, LevelMetrics] = field(default_factory=dict)

    @property
    def pass_at_k(self) -> float:
        """Overall Pass@k for this model."""
        return self.pass_at_k_sum / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def pass_at_1(self) -> float:
        """Overall Pass@1 for this model."""
        return self.pass_at_1_sum / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def syntax_pass_rate(self) -> float:
        """Overall syntax pass rate for this model."""
        return self.syntax_pass_sum / self.num_queries if self.num_queries > 0 else 0.0

    @property
    def avg_sharpe(self) -> Optional[float]:
        return float(np.mean(self.sharpes)) if self.sharpes else None

    @property
    def avg_total_return(self) -> Optional[float]:
        return float(np.mean(self.total_returns)) if self.total_returns else None

    @property
    def avg_annual_return(self) -> Optional[float]:
        return float(np.mean(self.annual_returns)) if self.annual_returns else None

    @property
    def avg_max_drawdown(self) -> Optional[float]:
        return float(np.mean(self.max_drawdowns)) if self.max_drawdowns else None

    @property
    def avg_win_rate(self) -> Optional[float]:
        return float(np.mean(self.win_rates)) if self.win_rates else None

    @property
    def avg_sortino(self) -> Optional[float]:
        return float(np.mean(self.sortinos)) if self.sortinos else None

    @property
    def avg_calmar(self) -> Optional[float]:
        return float(np.mean(self.calmars)) if self.calmars else None

    @property
    def avg_volatility(self) -> Optional[float]:
        return float(np.mean(self.volatilities)) if self.volatilities else None

    @property
    def avg_num_trades(self) -> Optional[float]:
        return float(np.mean(self.num_trades_list)) if self.num_trades_list else None

    @property
    def avg_execution_time_ms(self) -> Optional[float]:
        return float(np.mean(self.execution_times)) if self.execution_times else None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "num_queries": self.num_queries,
            "pass_at_k": self.pass_at_k,
            "pass_at_1": self.pass_at_1,
            "syntax_pass_rate": self.syntax_pass_rate,
            "avg_sharpe": self.avg_sharpe,
            "avg_total_return": self.avg_total_return,
            "avg_annual_return": self.avg_annual_return,
            "avg_max_drawdown": self.avg_max_drawdown,
            "avg_win_rate": self.avg_win_rate,
            "avg_sortino": self.avg_sortino,
            "avg_calmar": self.avg_calmar,
            "avg_volatility": self.avg_volatility,
            "avg_num_trades": self.avg_num_trades,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "by_level": {k: v.to_dict() for k, v in self.by_level.items()},
        }


class MetricsCalculator:
    """Calculate Pass@k and other metrics from backtest results."""

    def __init__(self, num_samples: int = 5):
        """
        Initialize metrics calculator.

        Args:
            num_samples: Number of samples per query (k in Pass@k)
        """
        self.num_samples = num_samples

    def calculate_query_metrics(
        self,
        results: List[BacktestResult],
        query_id: str,
        level: str,
    ) -> QueryMetrics:
        """
        Calculate metrics for a single query.

        Args:
            results: List of BacktestResult for this query
            query_id: Query identifier
            level: Difficulty level

        Returns:
            QueryMetrics object
        """
        metrics = QueryMetrics(query_id=query_id, level=level)

        # Sort by sample_id to ensure consistent ordering
        sorted_results = sorted(results, key=lambda r: r.sample_id)

        for i, result in enumerate(sorted_results):
            metrics.num_samples += 1

            if result.syntax_valid:
                metrics.num_syntax_valid += 1

            if result.passed:
                metrics.num_passed += 1

                # Collect all metrics from passed samples
                if result.sharpe is not None:
                    metrics.sharpes.append(result.sharpe)
                if result.total_return is not None:
                    metrics.total_returns.append(result.total_return)
                if result.annual_return is not None:
                    metrics.annual_returns.append(result.annual_return)
                if result.max_drawdown is not None:
                    metrics.max_drawdowns.append(result.max_drawdown)
                if result.win_rate is not None:
                    metrics.win_rates.append(result.win_rate)
                if result.sortino is not None:
                    metrics.sortinos.append(result.sortino)
                if result.calmar is not None:
                    metrics.calmars.append(result.calmar)
                if result.volatility is not None:
                    metrics.volatilities.append(result.volatility)
                if result.num_trades is not None:
                    metrics.num_trades_list.append(result.num_trades)
                if result.execution_time_ms is not None:
                    metrics.execution_times.append(result.execution_time_ms)

                # Check if first sample passed
                if i == 0:
                    metrics.first_sample_passed = True

        return metrics

    def calculate_model_metrics(
        self,
        results: List[BacktestResult],
        model: str,
    ) -> ModelMetrics:
        """
        Calculate aggregated metrics for a model.

        Args:
            results: All BacktestResult for this model
            model: Model name

        Returns:
            ModelMetrics object
        """
        model_metrics = ModelMetrics(model=model)

        # Group results by query_id
        by_query: Dict[str, List[BacktestResult]] = defaultdict(list)
        for result in results:
            by_query[result.query_id].append(result)

        # Calculate per-query metrics
        for query_id, query_results in by_query.items():
            # Extract level from query_id (e.g., "L1_easy_0" -> "L1_easy")
            parts = query_id.rsplit("_", 1)
            level = parts[0] if len(parts) > 1 else query_id

            query_metrics = self.calculate_query_metrics(
                query_results, query_id, level
            )

            # Update model-level aggregates
            model_metrics.num_queries += 1
            model_metrics.pass_at_k_sum += query_metrics.pass_at_k
            model_metrics.pass_at_1_sum += query_metrics.pass_at_1
            model_metrics.syntax_pass_sum += query_metrics.syntax_pass_rate
            model_metrics.sharpes.extend(query_metrics.sharpes)
            model_metrics.total_returns.extend(query_metrics.total_returns)
            model_metrics.annual_returns.extend(query_metrics.annual_returns)
            model_metrics.max_drawdowns.extend(query_metrics.max_drawdowns)
            model_metrics.win_rates.extend(query_metrics.win_rates)
            model_metrics.sortinos.extend(query_metrics.sortinos)
            model_metrics.calmars.extend(query_metrics.calmars)
            model_metrics.volatilities.extend(query_metrics.volatilities)
            model_metrics.num_trades_list.extend(query_metrics.num_trades_list)
            model_metrics.execution_times.extend(query_metrics.execution_times)

            # Update level-specific metrics
            if level not in model_metrics.by_level:
                model_metrics.by_level[level] = LevelMetrics(level=level)

            level_metrics = model_metrics.by_level[level]
            level_metrics.num_queries += 1
            level_metrics.pass_at_k_sum += query_metrics.pass_at_k
            level_metrics.pass_at_1_sum += query_metrics.pass_at_1
            level_metrics.syntax_pass_sum += query_metrics.syntax_pass_rate
            level_metrics.sharpes.extend(query_metrics.sharpes)
            level_metrics.total_returns.extend(query_metrics.total_returns)
            level_metrics.annual_returns.extend(query_metrics.annual_returns)
            level_metrics.max_drawdowns.extend(query_metrics.max_drawdowns)
            level_metrics.win_rates.extend(query_metrics.win_rates)
            level_metrics.sortinos.extend(query_metrics.sortinos)
            level_metrics.calmars.extend(query_metrics.calmars)
            level_metrics.volatilities.extend(query_metrics.volatilities)
            level_metrics.num_trades_list.extend(query_metrics.num_trades_list)
            level_metrics.execution_times.extend(query_metrics.execution_times)

        return model_metrics

    def calculate_all_metrics(
        self,
        results: List[BacktestResult],
        models: List[str],
    ) -> Dict[str, ModelMetrics]:
        """
        Calculate metrics for all models.

        Args:
            results: All BacktestResult objects
            models: List of model names

        Returns:
            Dict: {model: ModelMetrics}
        """
        all_metrics: Dict[str, ModelMetrics] = {}

        # Group results by model
        by_model: Dict[str, List[BacktestResult]] = defaultdict(list)
        for result in results:
            by_model[result.model].append(result)

        for model in models:
            model_results = by_model.get(model, [])
            model_metrics = self.calculate_model_metrics(model_results, model)
            all_metrics[model] = model_metrics

        return all_metrics

    def format_summary(self, all_metrics: Dict[str, ModelMetrics]) -> str:
        """
        Format metrics as a human-readable summary.

        Args:
            all_metrics: Output from calculate_all_metrics

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("LLM Benchmark Results Summary")
        lines.append("=" * 80)

        for model, metrics in all_metrics.items():
            lines.append(f"\n{model}")
            lines.append("-" * 60)

            lines.append(f"  Pass@{self.num_samples}: {metrics.pass_at_k:.2%}")
            lines.append(f"  Pass@1: {metrics.pass_at_1:.2%}")
            lines.append(f"  Syntax Pass Rate: {metrics.syntax_pass_rate:.2%}")
            lines.append(f"  Total Queries: {metrics.num_queries}")

            # Backtest metrics
            lines.append("  Backtest Metrics (avg across passed samples):")
            if metrics.avg_sharpe is not None:
                lines.append(f"    Sharpe: {metrics.avg_sharpe:.4f}")
            if metrics.avg_sortino is not None:
                lines.append(f"    Sortino: {metrics.avg_sortino:.4f}")
            if metrics.avg_calmar is not None:
                lines.append(f"    Calmar: {metrics.avg_calmar:.4f}")
            if metrics.avg_total_return is not None:
                lines.append(f"    Total Return: {metrics.avg_total_return:.4f}")
            if metrics.avg_annual_return is not None:
                lines.append(f"    Annual Return: {metrics.avg_annual_return:.4f}")
            if metrics.avg_max_drawdown is not None:
                lines.append(f"    Max Drawdown: {metrics.avg_max_drawdown:.4f}")
            if metrics.avg_volatility is not None:
                lines.append(f"    Volatility: {metrics.avg_volatility:.4f}")
            if metrics.avg_win_rate is not None:
                lines.append(f"    Win Rate: {metrics.avg_win_rate:.2%}")
            if metrics.avg_num_trades is not None:
                lines.append(f"    Avg Trades: {metrics.avg_num_trades:.1f}")
            if metrics.avg_execution_time_ms is not None:
                lines.append(f"    Avg Execution Time: {metrics.avg_execution_time_ms:.1f} ms")

            # Level breakdown
            if metrics.by_level:
                lines.append("  By Level:")
                for level, level_metrics in sorted(metrics.by_level.items()):
                    sharpe_str = f", Sharpe={level_metrics.avg_sharpe:.4f}" if level_metrics.avg_sharpe is not None else ""
                    lines.append(
                        f"    {level}: Pass@{self.num_samples}={level_metrics.pass_at_k:.2%}, "
                        f"Pass@1={level_metrics.pass_at_1:.2%}, "
                        f"n={level_metrics.num_queries}{sharpe_str}"
                    )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def to_json_dict(
        self, all_metrics: Dict[str, ModelMetrics]
    ) -> Dict[str, Any]:
        """
        Convert all metrics to JSON-serializable dict.

        Args:
            all_metrics: Output from calculate_all_metrics

        Returns:
            JSON-serializable dictionary
        """
        return {
            model: metrics.to_dict()
            for model, metrics in all_metrics.items()
        }
