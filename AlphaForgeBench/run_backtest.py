"""Standalone backtest script for existing samples."""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from AlphaForgeBench.backtest_runner import BacktestConfig, BacktestRunner
from AlphaForgeBench.code_extractor import CodeExtractor, CodeFixer
from AlphaForgeBench.factor_validator import FactorValidator
from AlphaForgeBench.metrics import MetricsCalculator


def _cleanup_and_exit():
    """清理所有子进程后强制退出。"""
    import os
    import signal

    current_pid = os.getpid()

    # 尝试杀死所有子进程
    try:
        import psutil
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except ImportError:
        # 如果没有 psutil，使用 pkill
        os.system(f"pkill -P {current_pid} 2>/dev/null")

    os._exit(0)


def load_samples_from_dir(
    samples_dir: Path,
    pass_count: int | None = None,
    processed_files: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load sample JSON files from directory.

    Args:
        samples_dir: Directory containing sample JSON files.
        pass_count: If set, only load samples with sample_id < pass_count.
        processed_files: Set of already processed file paths to skip.

    Returns:
        List of sample dictionaries.
    """
    samples = []
    processed_files = processed_files or set()

    for json_file in sorted(samples_dir.glob("*.json")):
        # Skip already processed files
        if str(json_file) in processed_files:
            continue

        try:
            with open(json_file, "r") as f:
                sample = json.load(f)
            # Ensure required fields exist
            if all(k in sample for k in ["query_id", "model", "sample_id", "response"]):
                # Filter by pass_count if specified
                if pass_count is not None:
                    sample_id = sample.get("sample_id", 0)
                    if sample_id >= pass_count:
                        continue
                sample["_file"] = str(json_file)
                samples.append(sample)
            else:
                print(f"Warning: Skipping {json_file.name} - missing required fields")
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping {json_file.name} - invalid JSON: {e}")
    return samples


def generate_output_dir(
    bench_dir: Path, pass_count: int | None, timestamp: str, symbol: str | None = None
) -> Path:
    """Generate output directory name with pass_count info.

    Args:
        bench_dir: Parent benchmark directory.
        pass_count: Pass count filter value.
        timestamp: Timestamp string for directory name.
        symbol: Optional symbol filter value.

    Returns:
        Path to output directory.
    """
    parts = ["result"]
    if pass_count is not None:
        parts.append(f"pass_{pass_count}")
    if symbol:
        parts.append(symbol)
    parts.append(timestamp)
    return bench_dir / "_".join(parts)


def analyze_samples(samples: list[dict]) -> dict[str, Any]:
    """Extract actual models/levels from samples."""
    models = set()
    query_levels = set()

    for sample in samples:
        models.add(sample["model"])
        # Extract level from query_id (e.g., "L1_hard_6" -> "L1_hard")
        query_id = sample["query_id"]
        parts = query_id.rsplit("_", 1)
        if len(parts) >= 1:
            level = parts[0]  # "L1_hard"
            query_levels.add(level)

    return {
        "models": sorted(models),
        "query_levels": sorted(query_levels),
    }


def load_config(config_path: Path | None, samples_dir: Path) -> dict[str, Any]:
    """Load benchmark config from specified path or auto-detect."""
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    # Try to find config in parent directory
    auto_config = samples_dir.parent / "benchmark_config.json"
    if auto_config.exists():
        print(f"Auto-detected config: {auto_config}")
        with open(auto_config, "r") as f:
            return json.load(f)

    # Return default config
    print("No config found, using defaults")
    return {
        "backtest": {
            "data_config": {
                "asset_name": "market",
                "data_type": "price",
                "level": "1day"
            },
            "start_ts": "2020-01-01 00:00:00",
            "end_ts": "2026-01-01 00:00:00",
            "history_ts": 300,
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_path": "datasets/market"
        }
    }


def clear_keys_from_cache(cache_file: Path, keys_to_remove: list[str]) -> int:
    """Remove specific keys from JSONL cache file.

    Args:
        cache_file: Path to the cache JSONL file.
        keys_to_remove: List of cache keys (or partial patterns) to remove.

    Returns:
        Number of entries removed.
    """
    if not cache_file.exists():
        return 0

    kept = []
    removed = 0
    with open(cache_file, "r") as f:
        for line in f:
            record = json.loads(line)
            key = record.get("_cache_key", "")
            should_remove = any(pattern in key for pattern in keys_to_remove)
            if should_remove:
                removed += 1
            else:
                kept.append(line)

    if removed > 0:
        with open(cache_file, "w") as f:
            f.writelines(kept)

    return removed


def clear_runtime_errors_from_cache(cache_file: Path) -> int:
    """Remove runtime error entries from JSONL cache file.

    Also removes 'not fully defined' syntax errors which are caused by
    Pydantic forward reference issues that can be fixed by system updates.

    Args:
        cache_file: Path to the cache JSONL file.

    Returns:
        Number of entries removed.
    """
    if not cache_file.exists():
        return 0

    kept = []
    removed = 0
    with open(cache_file, "r") as f:
        for line in f:
            record = json.loads(line)
            error_type = record.get("error_type")
            error_msg = str(record.get("error", ""))

            should_remove = (
                error_type == "runtime"
                or (error_type == "syntax" and "not fully defined" in error_msg)
            )

            if should_remove:
                removed += 1
            else:
                kept.append(line)

    if removed > 0:
        with open(cache_file, "w") as f:
            f.writelines(kept)

    return removed



def run_backtest(args: argparse.Namespace) -> None:
    """Main backtest execution."""
    from_extracted = bool(getattr(args, "extracted_codes", None))
    if from_extracted:
        extracted_path = Path(args.extracted_codes).resolve()
        if not extracted_path.exists():
            print(f"Error: extracted_codes file not found: {extracted_path}")
            return
        samples_dir = None
        bench_dir = extracted_path.parent
        config_ref = extracted_path  # load_config uses .parent -> the bench dir
    else:
        samples_dir = Path(args.samples_dir).resolve()
        if not samples_dir.exists():
            print(f"Error: Samples directory not found: {samples_dir}")
            return
        bench_dir = samples_dir.parent
        config_ref = samples_dir

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load config first to get default pass_count
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path, config_ref)

    # Get pass_count: from args, or default to config's num_samples
    pass_count = getattr(args, "pass_count", None)
    if pass_count is None:
        pass_count = config.get("num_samples")
        if pass_count is not None:
            print(f"Using num_samples from config as pass_count: {pass_count}")

    # Get symbols from config and validate --symbol if provided
    config_symbols = config["backtest"]["symbols"]
    if args.symbol:
        if args.symbol not in config_symbols:
            print(f"Error: Symbol '{args.symbol}' not in config symbols: {config_symbols}")
            return
        symbols = [args.symbol]
        print(f"Filtering to single symbol: {args.symbol}")
    else:
        symbols = config_symbols

    # Determine output directory: bench_xxx/result_{timestamp}/ or result_pass_N_{timestamp}/
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = generate_output_dir(bench_dir, pass_count, timestamp, args.symbol)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {extracted_path if from_extracted else samples_dir}")
    print(f"Output directory: {output_dir}")
    if pass_count is not None:
        print(f"Pass count filter: sample_id < {pass_count}")
    print()

    code_fixer = CodeFixer()
    if from_extracted:
        # Re-backtest an already-extracted answer set: skip sample loading + extraction.
        print("Loading extracted codes...")
        with open(extracted_path) as f:
            extracted_codes = json.load(f)
        if pass_count is not None:
            extracted_codes = [e for e in extracted_codes if e.get("sample_id", 0) < pass_count]
        for e in extracted_codes:
            e.setdefault("extraction_success", bool(e.get("strategy_code")))
        if not extracted_codes:
            print("Error: extracted_codes file is empty")
            return
        print(f"Loaded {len(extracted_codes)} extracted codes")
    else:
        # Load raw samples and extract code from their responses.
        print("Loading samples...")
        samples = load_samples_from_dir(samples_dir, pass_count=pass_count)
        if not samples:
            print("Error: No valid samples found")
            return
        print(f"Loaded {len(samples)} samples")

        print("Extracting code from responses...")
        extractor = CodeExtractor()
        extracted_codes = []
        for sample in samples:
            extracted = extractor.extract(sample["response"])

            # Apply code fixes (e.g., non-ASCII field names)
            fixed_strategy_code = extracted.strategy_code
            if extracted.strategy_code:
                fixed_strategy_code, _ = code_fixer.fix_code(
                    extracted.strategy_code,
                    query_id=sample["query_id"],
                    model=sample["model"],
                    sample_id=sample["sample_id"],
                )

            extracted_codes.append({
                "query_id": sample["query_id"],
                "model": sample["model"],
                "sample_id": sample["sample_id"],
                "query_text": sample.get("query_text", ""),
                "strategy_code": fixed_strategy_code,
                "factor_codes": extracted.factor_codes,
                "extraction_method": extracted.extraction_method,
                "extraction_success": extracted.has_strategy,
                "error": extracted.error,
            })

    # Analyze models / query levels (works on samples and extracted records alike)
    analysis = analyze_samples(extracted_codes)
    print(f"Models: {analysis['models']}")
    print(f"Query levels: {analysis['query_levels']}")
    success_count = sum(1 for e in extracted_codes if e.get("extraction_success"))
    print(f"Extraction success: {success_count}/{len(extracted_codes)}")
    print()

    # Save code fix report if there were any fixes
    fix_report = code_fixer.get_report()
    if fix_report.fixes:
        fix_report_path = output_dir / "invalid_code_report.json"
        fix_report.save(fix_report_path)
        print(f"Code fixes applied: {len(fix_report.fixes)} (saved to {fix_report_path.name})")

    # Save extracted codes
    extracted_codes_path = output_dir / "extracted_codes.json"
    with open(extracted_codes_path, "w") as f:
        json.dump(extracted_codes, f, indent=2, ensure_ascii=False)
    print(f"Saved: {extracted_codes_path}")
    print()

    # Create backtest config (needed for both factor validation and backtests)
    # Build data_configs from config (support both old single data_config and new data_configs format)
    if "data_configs" in config["backtest"]:
        data_configs = config["backtest"]["data_configs"]
    else:
        # Convert old single data_config to list with both price and feature
        base_config = config["backtest"]["data_config"]
        data_configs = [
            base_config,
            {
                "asset_name": base_config["asset_name"],
                "data_type": "feature",
                "level": base_config["level"],
            },
        ]

    backtest_config = BacktestConfig(
        data_configs=data_configs,
        start_ts=config["backtest"]["start_ts"],
        end_ts=config["backtest"]["end_ts"],
        history_ts=config["backtest"]["history_ts"],
        symbols=symbols,
        data_path=config["backtest"]["data_path"],
    )

    # Validate and compute missing factors
    print("Validating factors and computing missing ones...")
    allow_compute = getattr(args, "allow_compute_factors", False)
    factor_validator = FactorValidator(
        backtest_config,
        strict_mode=not allow_compute,
        use_auto_meta=allow_compute,
    )
    if not allow_compute:
        print("Strict mode: factors not in meta_info.json will be treated as errors")
        print("Use --allow-compute-factors to enable automatic factor computation")
    validation_result = asyncio.run(factor_validator.validate_and_compute(extracted_codes))

    # Save invalid factors report
    invalid_report_path = output_dir / "invalid_factors_report.json"
    factor_validator.save_invalid_report(validation_result, invalid_report_path)
    print(f"Saved: {invalid_report_path}")

    if validation_result.computed_factors:
        print(f"Computed {len(validation_result.computed_factors)} new factors: {validation_result.computed_factors}")
    if validation_result.invalid_factors:
        invalid_names = [inv.factor_name for inv in validation_result.invalid_factors]
        print(f"Warning: Found {len(validation_result.invalid_factors)} invalid factors: {invalid_names}")
    print()

    # Run backtests
    print("Running backtests...")

    # Use shared cache directory (at bench level, not result level)
    cache_dir = bench_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "backtest_cache.json"

    # Clear all cache if requested
    if getattr(args, "clear_cache", False):
        if cache_file.exists():
            cache_file.unlink()
            print("Cleared all cache entries")

    # Clear runtime errors from cache if requested
    if getattr(args, "retry_errors", False):
        removed = clear_runtime_errors_from_cache(cache_file)
        if removed > 0:
            print(f"Cleared {removed} runtime error entries from cache")

    # Clear specific keys if requested
    if getattr(args, "retry_keys", None):
        removed = clear_keys_from_cache(cache_file, args.retry_keys)
        if removed > 0:
            print(f"Cleared {removed} entries matching: {args.retry_keys}")

    runner = BacktestRunner(backtest_config)
    runner.set_cache_file(cache_file)
    num_workers = getattr(args, "workers", 1)
    timeout = getattr(args, "timeout", 10800)
    if num_workers > 1:
        print(f"Using {num_workers} parallel workers, timeout: {timeout}s")
    backtest_results_raw = runner.run_batch_backtest(extracted_codes, num_workers=num_workers, timeout=timeout)
    backtest_results = [r.to_dict() for r in backtest_results_raw]

    # Save backtest results
    backtest_results_path = output_dir / "backtest_results.json"
    with open(backtest_results_path, "w") as f:
        json.dump(backtest_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {backtest_results_path}")
    print()

    # Calculate metrics
    print("Calculating metrics...")

    # Get num_samples from config or infer from data
    num_samples = config.get("num_samples", 1)
    # Try to infer from actual data
    sample_ids = set()
    for rec in extracted_codes:
        sample_ids.add(rec["sample_id"])
    if sample_ids:
        num_samples = max(num_samples, max(sample_ids) + 1)

    calculator = MetricsCalculator(num_samples=num_samples)

    # calculate_all_metrics expects BacktestResult objects
    all_metrics = calculator.calculate_all_metrics(
        results=backtest_results_raw,
        models=analysis["models"],
    )

    # Convert to serializable format
    metrics_dict = {}
    for model, model_metrics in all_metrics.items():
        metrics_dict[model] = model_metrics.to_dict()

    # Build final results
    results = {
        "timestamp": datetime.now().isoformat(),
        "samples_dir": str(samples_dir) if samples_dir else str(extracted_path),
        "config": config,
        "pass_count": pass_count,
        "pass_count_description": (
            f"Only sample_id < {pass_count}" if pass_count else "All samples"
        ),
        "analysis": analysis,
        "num_samples_detected": num_samples,
        "extraction_stats": {
            "total": len(extracted_codes),
            "success": success_count,
            "failed": len(extracted_codes) - success_count,
        },
        "metrics": metrics_dict,
        "summary": build_summary(metrics_dict, analysis),
    }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {results_path}")
    print()

    # Print summary
    print_summary(results)

    # 清理所有子进程后退出
    _cleanup_and_exit()


def build_summary(metrics_dict: dict, analysis: dict) -> dict:
    """Build summary statistics from metrics_dict.

    metrics_dict structure: {model: ModelMetrics.to_dict()}
    """
    summary = {
        "by_model": {},
        "by_level": {},
        "overall": {},
    }

    all_pass_at_k = []
    all_pass_at_1 = []
    all_syntax_pass = []

    # Aggregate by model
    for model in analysis["models"]:
        if model not in metrics_dict:
            continue
        m = metrics_dict[model]
        if m.get("num_queries", 0) > 0:
            summary["by_model"][model] = {
                "pass_at_k": m.get("pass_at_k", 0),
                "pass_at_1": m.get("pass_at_1", 0),
                "syntax_pass_rate": m.get("syntax_pass_rate", 0),
            }
            all_pass_at_k.append(m.get("pass_at_k", 0))
            all_pass_at_1.append(m.get("pass_at_1", 0))
            all_syntax_pass.append(m.get("syntax_pass_rate", 0))

    # Aggregate by level
    for level in analysis["query_levels"]:
        level_pass_at_k = []
        level_pass_at_1 = []
        level_syntax_pass = []

        for model in analysis["models"]:
            if model not in metrics_dict:
                continue
            m = metrics_dict[model]
            by_level = m.get("by_level", {})
            if level in by_level:
                lm = by_level[level]
                if lm.get("num_queries", 0) > 0:
                    level_pass_at_k.append(lm.get("pass_at_k", 0))
                    level_pass_at_1.append(lm.get("pass_at_1", 0))
                    level_syntax_pass.append(lm.get("syntax_pass_rate", 0))

        if level_pass_at_k:
            summary["by_level"][level] = {
                "pass_at_k": sum(level_pass_at_k) / len(level_pass_at_k),
                "pass_at_1": sum(level_pass_at_1) / len(level_pass_at_1),
                "syntax_pass_rate": sum(level_syntax_pass) / len(level_syntax_pass),
            }

    # Overall
    if all_pass_at_k:
        summary["overall"] = {
            "pass_at_k": sum(all_pass_at_k) / len(all_pass_at_k),
            "pass_at_1": sum(all_pass_at_1) / len(all_pass_at_1),
            "syntax_pass_rate": sum(all_syntax_pass) / len(all_syntax_pass),
        }

    return summary


def print_summary(results: dict) -> None:
    """Print summary table to console."""
    print("=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    summary = results.get("summary", {})

    # Overall
    overall = summary.get("overall", {})
    if overall:
        print("\nOverall:")
        print(f"  pass@k: {overall.get('pass_at_k', 0):.4f}")
        print(f"  pass@1: {overall.get('pass_at_1', 0):.4f}")
        print(f"  syntax_pass_rate: {overall.get('syntax_pass_rate', 0):.4f}")

    # By model
    by_model = summary.get("by_model", {})
    if by_model:
        print("\nBy Model:")
        for model, stats in by_model.items():
            print(f"  {model}:")
            print(f"    pass@k={stats.get('pass_at_k', 0):.4f}, pass@1={stats.get('pass_at_1', 0):.4f}, syntax={stats.get('syntax_pass_rate', 0):.4f}")

    # By level
    by_level = summary.get("by_level", {})
    if by_level:
        print("\nBy Level:")
        for level, stats in sorted(by_level.items()):
            print(f"  {level}: pass@k={stats.get('pass_at_k', 0):.4f}, pass@1={stats.get('pass_at_1', 0):.4f}")

    print()
    print("=" * 60)


def run_backtest_watch(args: argparse.Namespace) -> None:
    """Run backtest in watch mode - continuously monitor for new samples."""
    samples_dir = Path(args.samples_dir).resolve()
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        return

    timeout_seconds = getattr(args, "watch_timeout", 180)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = samples_dir.parent

    # Load config first to get default pass_count
    config_path = args.config
    if not config_path:
        config_path = bench_dir / "benchmark_config.json"
        if not config_path.exists():
            config_path = samples_dir / "benchmark_config.json"

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        print("Warning: No config file found, using defaults")
        config = {
            "backtest": {
                "start_ts": "2020-01-01 00:00:00",
                "end_ts": "2026-01-01 00:00:00",
                "history_ts": 120,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "data_path": "datasets/market",
            },
            "data": {
                "asset_name": "market",
                "level": "1day",
            },
        }

    # Get pass_count: from args, or default to config's num_samples
    pass_count = getattr(args, "pass_count", None)
    if pass_count is None:
        pass_count = config.get("num_samples")
        if pass_count is not None:
            print(f"Using num_samples from config as pass_count: {pass_count}")

    # Get symbols from config and validate --symbol if provided
    config_symbols = config["backtest"]["symbols"]
    if args.symbol:
        if args.symbol not in config_symbols:
            print(f"Error: Symbol '{args.symbol}' not in config symbols: {config_symbols}")
            return
        symbols = [args.symbol]
        print(f"Filtering to single symbol: {args.symbol}")
    else:
        symbols = config_symbols

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = generate_output_dir(bench_dir, pass_count, timestamp, args.symbol)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WATCH MODE STARTED")
    print("=" * 60)
    print(f"Samples directory: {samples_dir}")
    print(f"Output directory: {output_dir}")
    if pass_count is not None:
        print(f"Pass count filter: sample_id < {pass_count}")
    print(f"Idle timeout: {timeout_seconds} seconds")
    print()

    # Track processed files and accumulated results
    processed_files: set[str] = set()
    all_extracted_codes: list[dict] = []
    all_backtest_results: list = []
    last_activity_time = time.time()

    # Setup backtest runner
    base_config = config.get("data", {})
    data_configs = config.get("data_configs")
    if not data_configs:
        data_configs = [
            {
                "asset_name": base_config.get("asset_name", "market"),
                "data_type": "price",
                "level": base_config.get("level", "1day"),
            },
            {
                "asset_name": base_config.get("asset_name", "market"),
                "data_type": "feature",
                "level": base_config.get("level", "1day"),
            },
        ]

    backtest_config = BacktestConfig(
        data_configs=data_configs,
        start_ts=config["backtest"]["start_ts"],
        end_ts=config["backtest"]["end_ts"],
        history_ts=config["backtest"]["history_ts"],
        symbols=symbols,
        data_path=config["backtest"]["data_path"],
    )

    # Use shared cache directory
    cache_dir = bench_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "backtest_cache.json"

    # Clear all cache if requested
    if getattr(args, "clear_cache", False):
        if cache_file.exists():
            cache_file.unlink()
            print("Cleared all cache entries")

    # Clear runtime errors from cache if requested
    if getattr(args, "retry_errors", False):
        removed = clear_runtime_errors_from_cache(cache_file)
        if removed > 0:
            print(f"Cleared {removed} runtime error entries from cache")

    # Clear specific keys if requested
    if getattr(args, "retry_keys", None):
        removed = clear_keys_from_cache(cache_file, args.retry_keys)
        if removed > 0:
            print(f"Cleared {removed} entries matching: {args.retry_keys}")

    runner = BacktestRunner(backtest_config)
    runner.set_cache_file(cache_file)

    extractor = CodeExtractor()
    code_fixer = CodeFixer()
    allow_compute = getattr(args, "allow_compute_factors", False)
    factor_validator = FactorValidator(
        backtest_config,
        strict_mode=not allow_compute,
        use_auto_meta=allow_compute,
    )
    if not allow_compute:
        print("Strict mode: factors not in meta_info.json will be treated as errors")
        print("Use --allow-compute-factors to enable automatic factor computation")

    try:
        while True:
            # Load new samples
            new_samples = load_samples_from_dir(
                samples_dir, pass_count=pass_count, processed_files=processed_files
            )

            if new_samples:
                last_activity_time = time.time()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(new_samples)} new samples")

                # Mark files as processed
                for sample in new_samples:
                    processed_files.add(sample["_file"])

                # Extract code from new samples
                new_extracted = []
                for sample in new_samples:
                    extracted = extractor.extract(sample["response"])

                    # Apply code fixes (e.g., non-ASCII field names)
                    fixed_strategy_code = extracted.strategy_code
                    if extracted.strategy_code:
                        fixed_strategy_code, _ = code_fixer.fix_code(
                            extracted.strategy_code,
                            query_id=sample["query_id"],
                            model=sample["model"],
                            sample_id=sample["sample_id"],
                        )

                    new_extracted.append({
                        "query_id": sample["query_id"],
                        "model": sample["model"],
                        "sample_id": sample["sample_id"],
                        "query_text": sample.get("query_text", ""),
                        "strategy_code": fixed_strategy_code,
                        "factor_codes": extracted.factor_codes,
                        "extraction_method": extracted.extraction_method,
                        "extraction_success": extracted.has_strategy,
                        "error": extracted.error,
                    })

                all_extracted_codes.extend(new_extracted)

                # Validate factors
                validation_result = asyncio.run(
                    factor_validator.validate_and_compute(new_extracted)
                )

                # Run backtests for new samples
                num_workers = getattr(args, "workers", 1)
                timeout = getattr(args, "timeout", 10800)
                new_backtest_results = runner.run_batch_backtest(new_extracted, num_workers=num_workers, timeout=timeout)
                all_backtest_results.extend(new_backtest_results)

                print(f"Processed {len(new_samples)} samples. Total: {len(all_extracted_codes)}")

            else:
                # No new samples, check timeout
                idle_time = time.time() - last_activity_time
                if idle_time >= timeout_seconds:
                    print(f"\n\nNo new samples for {timeout_seconds}s. Exiting watch mode.")
                    break

                # Show waiting status
                remaining = int(timeout_seconds - idle_time)
                print(
                    f"\r[{datetime.now().strftime('%H:%M:%S')}] Waiting for new samples... "
                    f"({remaining}s until exit, {len(all_extracted_codes)} processed)",
                    end="",
                    flush=True,
                )
                time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving current progress...")

    # Finalize and save results
    if all_extracted_codes:
        print("\n\nFinalizing results...")

        # Analyze all samples
        all_samples = []
        for ec in all_extracted_codes:
            all_samples.append({
                "query_id": ec["query_id"],
                "model": ec["model"],
                "sample_id": ec["sample_id"],
            })
        analysis = analyze_samples(all_samples)

        # Calculate metrics
        num_samples = config.get("num_samples", 1)
        sample_ids = {ec["sample_id"] for ec in all_extracted_codes}
        if sample_ids:
            num_samples = max(num_samples, max(sample_ids) + 1)

        calculator = MetricsCalculator(num_samples=num_samples)
        all_metrics = calculator.calculate_all_metrics(
            results=all_backtest_results,
            models=analysis["models"],
        )

        # Convert to serializable format
        metrics_dict = {}
        for model, model_metrics in all_metrics.items():
            metrics_dict[model] = model_metrics.to_dict()

        # Count successes
        success_count = sum(1 for ec in all_extracted_codes if ec["extraction_success"])

        # Build final results
        results = {
            "timestamp": datetime.now().isoformat(),
            "samples_dir": str(samples_dir),
            "config": config,
            "pass_count": pass_count,
            "pass_count_description": (
                f"Only sample_id < {pass_count}" if pass_count else "All samples"
            ),
            "watch_mode": True,
            "watch_timeout_seconds": timeout_seconds,
            "analysis": analysis,
            "num_samples_detected": num_samples,
            "extraction_stats": {
                "total": len(all_extracted_codes),
                "success": success_count,
                "failed": len(all_extracted_codes) - success_count,
            },
            "metrics": metrics_dict,
            "summary": build_summary(metrics_dict, analysis),
        }

        # Save results
        print("Saving results...", flush=True)
        results_path = output_dir / "results.json"
        print(f"  → results.json...", end=" ", flush=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("done")

        # Save backtest results
        backtest_results_path = output_dir / "backtest_results.json"
        print(f"  → backtest_results.json ({len(all_backtest_results)} items)...", end=" ", flush=True)
        with open(backtest_results_path, "w") as f:
            json.dump([r.to_dict() for r in all_backtest_results], f, indent=2, ensure_ascii=False)
        print("done")

        # Save extracted codes
        extracted_codes_path = output_dir / "extracted_codes.json"
        print(f"  → extracted_codes.json...", end=" ", flush=True)
        with open(extracted_codes_path, "w") as f:
            json.dump(all_extracted_codes, f, indent=2, ensure_ascii=False)
        print("done")

        # Save invalid factors report
        invalid_report_path = output_dir / "invalid_factors_report.json"
        factor_validator.save_invalid_report(validation_result, invalid_report_path)
        print(f"Saved: {invalid_report_path}")

        # Save code fix report if there were any fixes
        fix_report = code_fixer.get_report()
        if fix_report.fixes:
            fix_report_path = output_dir / "invalid_code_report.json"
            fix_report.save(fix_report_path)
            print(f"Code fixes applied: {len(fix_report.fixes)} (saved to {fix_report_path.name})")

        print()
        print_summary(results)

        # 清理所有子进程后退出
        _cleanup_and_exit()
    else:
        print("\nNo samples were processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone backtest script for existing samples"
    )
    parser.add_argument(
        "--samples-dir",
        default=None,
        help="Path to a samples directory (raw per-sample JSON files from a benchmark run)",
    )
    parser.add_argument(
        "--extracted-codes",
        default=None,
        help="Path to an extracted_codes.json (already-parsed strategy/factor code). Re-backtests "
             "that answer set directly, skipping sample loading + code extraction. Exactly one of "
             "--samples-dir / --extracted-codes is required.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: samples parent directory)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to benchmark_config.json (auto-detected if not specified)",
    )
    parser.add_argument(
        "--pass-count",
        type=int,
        default=None,
        help="Only process samples with sample_id < N (e.g., --pass-count 1 for sample_id=0 only)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode: continuously monitor for new samples",
    )
    parser.add_argument(
        "--watch-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for watch mode (default: 180 = 3 minutes)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Remove runtime error entries from cache before running (retry failed backtests)",
    )
    parser.add_argument(
        "--retry-keys",
        type=str,
        nargs="+",
        default=None,
        help="Only retry specific cache keys (patterns). E.g., --retry-keys L2_medium_28 L3_hard_27",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Only run backtest for this symbol (must be in config's symbols list)",
    )
    parser.add_argument(
        "--allow-compute-factors",
        action="store_true",
        help="Allow computing/using factors not in meta_info.json. If not set, missing factors will be treated as errors.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cache entries before running backtests (start fresh)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for backtesting (default: 1 = serial execution)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10800,
        help="Timeout in seconds for batch backtest (default: 10800 = 3 hours)",
    )

    args = parser.parse_args()

    if not args.samples_dir and not args.extracted_codes:
        parser.error("one of --samples-dir / --extracted-codes is required")
    if args.extracted_codes and args.watch:
        parser.error("--extracted-codes cannot be combined with --watch (watch needs a samples dir)")

    if args.watch:
        run_backtest_watch(args)
    else:
        run_backtest(args)


if __name__ == "__main__":
    main()
