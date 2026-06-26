"""Standalone backtest script with priority ordering.

This script prioritizes getting results for each model/level/difficulty combination first,
so you can see initial results across all combinations before all samples are processed.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from AlphaForgeBench.backtest_runner import BacktestConfig, BacktestRunner
from AlphaForgeBench.code_extractor import CodeExtractor
from AlphaForgeBench.factor_validator import FactorValidator
from AlphaForgeBench.metrics import MetricsCalculator


def load_samples_from_dir(samples_dir: Path) -> list[dict[str, Any]]:
    """Load all sample JSON files from directory."""
    samples = []
    for json_file in sorted(samples_dir.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                sample = json.load(f)
            # Ensure required fields exist
            if all(k in sample for k in ["query_id", "model", "sample_id", "response"]):
                sample["_file"] = str(json_file)
                samples.append(sample)
            else:
                print(f"Warning: Skipping {json_file.name} - missing required fields")
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping {json_file.name} - invalid JSON: {e}")
    return samples


def sort_samples_by_priority(samples: list[dict]) -> list[dict]:
    """
    Sort samples to prioritize getting results for each model/level/difficulty combination first.

    Priority order: sample_id -> query_index -> model -> level -> difficulty
    This ensures we get one result for each of the 9 level×difficulty combinations per model first.

    Example order:
    ModelA_L1_easy_0_sample0, ModelA_L1_medium_0_sample0, ModelA_L1_hard_0_sample0,
    ModelA_L2_easy_0_sample0, ModelA_L2_medium_0_sample0, ModelA_L2_hard_0_sample0,
    ModelA_L3_easy_0_sample0, ModelA_L3_medium_0_sample0, ModelA_L3_hard_0_sample0,
    ModelB_L1_easy_0_sample0, ...
    """
    def sort_key(sample):
        model = sample["model"]
        query_id = sample["query_id"]  # e.g., "L1_easy_0"
        sample_id = sample["sample_id"]

        # Parse query_id to extract level, difficulty, query_index
        # Format: L{level}_{difficulty}_{index}
        parts = query_id.split("_")
        level = parts[0] if len(parts) > 0 else "L1"  # "L1", "L2", "L3"
        difficulty = parts[1] if len(parts) > 1 else "easy"  # "easy", "medium", "hard"
        query_index = int(parts[2]) if len(parts) > 2 else 0

        # Level order: L1 -> L2 -> L3
        level_order = {"L1": 0, "L2": 1, "L3": 2}

        # Difficulty order: easy -> medium -> hard
        diff_order = {"easy": 0, "medium": 1, "hard": 2}

        # Priority: sample_id first, then query_index, then model, then level, then difficulty
        return (sample_id, query_index, model, level_order.get(level, 0), diff_order.get(difficulty, 0))

    return sorted(samples, key=sort_key)


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


def run_backtest(args: argparse.Namespace) -> None:
    """Main backtest execution."""
    samples_dir = Path(args.samples_dir).resolve()
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        return

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output directory: bench_xxx/result_{timestamp}/
    bench_dir = samples_dir.parent
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = bench_dir / f"result_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Samples directory: {samples_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load samples
    print("Loading samples...")
    samples = load_samples_from_dir(samples_dir)
    if not samples:
        print("Error: No valid samples found")
        return
    print(f"Loaded {len(samples)} samples")

    # Sort samples by priority
    print("Sorting samples by priority (sample_id -> model -> level -> difficulty)...")
    samples = sort_samples_by_priority(samples)
    print(f"First 5 samples after sorting:")
    for i, s in enumerate(samples[:5]):
        print(f"  {i+1}. {s['model'].split('/')[-1]} | {s['query_id']} | sample_{s['sample_id']}")
    print()

    # Analyze samples to get actual models/levels
    analysis = analyze_samples(samples)
    print(f"Models: {analysis['models']}")
    print(f"Query levels: {analysis['query_levels']}")
    print()

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path, samples_dir)

    # Extract code from responses
    print("Extracting code from responses...")
    extractor = CodeExtractor()
    extracted_codes = []

    for sample in samples:
        extracted = extractor.extract(sample["response"])
        extracted_codes.append({
            "query_id": sample["query_id"],
            "model": sample["model"],
            "sample_id": sample["sample_id"],
            "query_text": sample.get("query_text", ""),
            "strategy_code": extracted.strategy_code,
            "factor_codes": extracted.factor_codes,
            "extraction_method": extracted.extraction_method,
            "extraction_success": extracted.has_strategy,
            "error": extracted.error,
        })

    success_count = sum(1 for e in extracted_codes if e["extraction_success"])
    print(f"Extraction success: {success_count}/{len(extracted_codes)}")

    # Save extracted codes
    extracted_codes_path = output_dir / "extracted_codes.json"
    with open(extracted_codes_path, "w") as f:
        json.dump(extracted_codes, f, indent=2)
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
        symbols=config["backtest"]["symbols"],
        data_path=config["backtest"]["data_path"],
    )

    # Validate and compute missing factors
    print("Validating factors and computing missing ones...")
    factor_validator = FactorValidator(backtest_config)
    validation_result = asyncio.run(factor_validator.validate_and_compute(extracted_codes))

    # Save invalid factors report
    invalid_report_path = output_dir / "invalid_factors_report.json"
    factor_validator.save_invalid_report(validation_result, invalid_report_path)
    print(f"Saved: {invalid_report_path}")

    if validation_result.computed_factors:
        print(f"Computed {len(validation_result.computed_factors)} new factors: {validation_result.computed_factors}")
    if validation_result.invalid_factors:
        print(f"Warning: Found {len(validation_result.invalid_factors)} invalid factors")
    print()

    # Run backtests
    print("Running backtests...")

    # Use shared cache directory (at bench level, not result level)
    cache_dir = bench_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    runner = BacktestRunner(backtest_config)
    runner.set_cache_file(cache_dir / "backtest_cache.json")
    backtest_results_raw = runner.run_batch_backtest(extracted_codes)
    backtest_results = [r.to_dict() for r in backtest_results_raw]

    # Save backtest results
    backtest_results_path = output_dir / "backtest_results.json"
    with open(backtest_results_path, "w") as f:
        json.dump(backtest_results, f, indent=2)
    print(f"Saved: {backtest_results_path}")
    print()

    # Calculate metrics
    print("Calculating metrics...")

    # Get num_samples from config or infer from data
    num_samples = config.get("num_samples", 1)
    # Try to infer from actual data
    sample_ids = set()
    for sample in samples:
        sample_ids.add(sample["sample_id"])
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
        "samples_dir": str(samples_dir),
        "config": config,
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
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    print()

    # Print summary
    print_summary(results)


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


def main():
    parser = argparse.ArgumentParser(
        description="Standalone backtest script with priority ordering"
    )
    parser.add_argument(
        "--samples-dir",
        required=True,
        help="Path to samples directory containing JSON files",
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

    args = parser.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
