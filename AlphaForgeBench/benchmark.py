#!/usr/bin/env python3
"""
LLM Benchmark - Main entry point for evaluating LLM code generation quality.

Usage:
    # 使用配置文件运行 (推荐)
    uv run python -m AlphaForgeBench.benchmark

    # 使用自定义配置文件
    uv run python -m AlphaForgeBench.benchmark --config path/to/config.json

    # 命令行参数覆盖配置文件
    uv run python -m AlphaForgeBench.benchmark \
        --models "openrouter/gpt-5.2,openrouter/claude-sonnet-4.5" \
        --samples 5

    # 查看结果
    cat AlphaForgeBench/benchmark_results/latest/results.json | jq '.results'
"""

import argparse
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from AlphaForgeBench.config import (
    BenchmarkConfig,
    DEFAULT_CONFIG_PATH,
    QUERY_LEVELS,
)
from AlphaForgeBench.api_caller import (
    LLMCaller,
    SampleResult,
    load_queries_from_checkpoint,
)
from AlphaForgeBench.code_extractor import CodeExtractor
from AlphaForgeBench.backtest_runner import BacktestRunner, BacktestConfig, BacktestResult
from AlphaForgeBench.metrics import MetricsCalculator
from AlphaForgeBench.factor_validator import FactorValidator, ValidationResult

from src.logger import logger

console = Console()


class LLMBenchmark:
    """Main benchmark orchestrator."""

    def __init__(self, config: BenchmarkConfig, resume: Optional[str] = None, skip_backtest: bool = False, retry_failed: bool = False, strict: bool = False):
        self.config = config
        self.resume = resume  # "auto" or specific benchmark_id
        self.skip_backtest = skip_backtest
        self.retry_failed = retry_failed
        # Factor validation mode. Default = permissive (compute missing factors, use auto meta)
        # to match the published/paper results; pass --strict for base-meta-only validation.
        self.strict = strict
        self.caller: Optional[LLMCaller] = None
        self.extractor = CodeExtractor()
        self.runner: Optional[BacktestRunner] = None
        self.metrics_calc = MetricsCalculator(num_samples=config.num_samples)

        # Factor validator (initialized in setup)
        self.validator: Optional[FactorValidator] = None

        # Results storage
        self.sample_results: List[SampleResult] = []
        self.extracted_codes: List[Dict[str, Any]] = []
        self.validation_result: Optional[ValidationResult] = None
        self.backtest_results: List[BacktestResult] = []

    def _compare_system_prompt(self, bench_dir: Path) -> bool:
        """
        Compare current system prompt with saved one in benchmark directory.

        Args:
            bench_dir: Benchmark directory path

        Returns:
            True if system prompts match, False otherwise
        """
        saved_prompt_path = bench_dir / "system_prompt.txt"
        if not saved_prompt_path.exists():
            logger.warning(f"No saved system prompt in {bench_dir}")
            return False

        if not self.config.system_prompt_path.exists():
            logger.warning(f"Current system prompt not found: {self.config.system_prompt_path}")
            return False

        try:
            with open(saved_prompt_path, "r", encoding="utf-8") as f:
                saved_content = f.read()
            with open(self.config.system_prompt_path, "r", encoding="utf-8") as f:
                current_content = f.read()

            if saved_content != current_content:
                logger.warning(
                    f"System prompt content differs from saved version in {bench_dir.name}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Failed to compare system prompts: {e}")
            return False

    def _count_existing_samples(self) -> Dict[str, int]:
        """
        Count existing samples per model in the samples directory.

        Returns:
            Dict mapping model name to max sample_id found + 1
        """
        samples_dir = self.config.samples_dir
        if not samples_dir.exists():
            return {}

        # Count samples per model
        model_samples: Dict[str, set] = {}

        for json_file in samples_dir.glob("*.json"):
            try:
                # Parse filename: {model}_{query_type}_{query_id}_{sample_id}.json
                parts = json_file.stem.rsplit("_", 1)
                if len(parts) == 2:
                    sample_id = int(parts[1])
                    # Extract model from the beginning
                    model_part = parts[0].rsplit("_", 2)[0]  # Remove query_type and query_id
                    if model_part not in model_samples:
                        model_samples[model_part] = set()
                    model_samples[model_part].add(sample_id)
            except (ValueError, IndexError):
                continue

        # Return max sample count per model
        return {model: max(samples) + 1 for model, samples in model_samples.items() if samples}

    def find_compatible_benchmark(self) -> Optional[Path]:
        """
        Find a compatible existing benchmark directory.

        Returns:
            Path to compatible benchmark directory, or None if not found
        """
        if not self.config.results_dir.exists():
            return None

        # Get all benchmark directories
        bench_dirs = [
            d for d in self.config.results_dir.iterdir()
            if d.is_dir() and d.name.startswith("bench_")
        ]

        if not bench_dirs:
            return None

        # Sort by modification time (newest first)
        bench_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        for bench_dir in bench_dirs:
            config_file = bench_dir / "benchmark_config.json"
            if not config_file.exists():
                continue

            try:
                saved_config = BenchmarkConfig.from_saved_config(config_file)
                if self.config.is_compatible_with(saved_config):
                    # Also check system prompt content
                    if self._compare_system_prompt(bench_dir):
                        return bench_dir
                    else:
                        logger.info(f"Skipping {bench_dir.name}: system prompt content differs")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                continue

        return None

    def setup(self):
        """Initialize all components."""
        # Handle resume logic
        existing_dir = None
        if self.resume:
            if self.resume == "auto":
                existing_dir = self.find_compatible_benchmark()
                if existing_dir:
                    logger.info(f"Found compatible benchmark: {existing_dir.name}")
            else:
                # Specific benchmark ID provided - must be compatible or fail
                specific_dir = self.config.results_dir / self.resume
                if not specific_dir.exists():
                    raise ValueError(f"Specified benchmark directory not found: {self.resume}")

                config_file = specific_dir / "benchmark_config.json"
                if not config_file.exists():
                    raise ValueError(f"No benchmark_config.json found in: {self.resume}")

                saved_config = BenchmarkConfig.from_saved_config(config_file)

                # Check temperature specifically for clear error message
                if self.config.temperature != saved_config.temperature:
                    raise ValueError(
                        f"Temperature mismatch! Current: {self.config.temperature}, "
                        f"Saved: {saved_config.temperature}. "
                        f"Cannot resume with different temperature."
                    )

                # Check full compatibility
                if not self.config.is_compatible_with(saved_config):
                    raise ValueError(
                        f"Config incompatible with {self.resume}. "
                        f"Current: {self.config.get_compatible_config_dict()}, "
                        f"Saved: {saved_config.get_compatible_config_dict()}"
                    )

                # Check system prompt content
                if not self._compare_system_prompt(specific_dir):
                    raise ValueError(
                        f"System prompt content differs from saved benchmark {self.resume}. "
                        f"Cannot resume with different system prompt."
                    )

                existing_dir = specific_dir
                logger.info(f"Resuming from specified benchmark: {self.resume}")

        if existing_dir:
            # Reuse existing directory
            self.config.benchmark_id = existing_dir.name
            logger.info(f"Resuming benchmark: {self.config.benchmark_id}")

            # Check existing samples count and show incremental info
            existing_samples = self._count_existing_samples()
            if existing_samples:
                logger.info(f"Found existing samples: {existing_samples}")
                logger.info(f"Target num_samples: {self.config.num_samples}")
        else:
            if self.resume:
                logger.info("No compatible benchmark found, creating new one")

        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.samples_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save benchmark config (always update with current models)
        config_path = self.config.output_dir / "benchmark_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

        # Copy system prompt file
        if self.config.system_prompt_path.exists():
            system_prompt_dest = self.config.output_dir / "system_prompt.txt"
            if not system_prompt_dest.exists():
                shutil.copy2(self.config.system_prompt_path, system_prompt_dest)

        # Initialize LLM caller
        if not self.config.system_prompt:
            raise ValueError("System prompt is required")

        self.caller = LLMCaller(
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_concurrent=self.config.max_concurrent_requests,
            retry_failed=self.retry_failed,
        )

        # Initialize backtest runner
        # Convert single data_config to data_configs list format
        data_configs = [
            self.config.data_config,
            {
                "asset_name": self.config.data_config.get("asset_name", "market"),
                "data_type": "feature",
                "level": self.config.data_config.get("level", "1day"),
            },
        ]
        backtest_config = BacktestConfig(
            data_path=self.config.data_path,
            data_configs=data_configs,
            start_ts=self.config.start_ts,
            end_ts=self.config.end_ts,
            history_ts=self.config.history_ts,
            symbols=self.config.symbols,
        )
        self.runner = BacktestRunner(backtest_config)
        self.runner.set_cache_file(self.config.cache_dir / "backtest_cache.json")

        # Initialize factor validator (permissive by default — matches the paper; --strict flips it)
        self.validator = FactorValidator(self.config, strict_mode=self.strict, use_auto_meta=not self.strict)

        # Create symlink to latest
        latest_link = self.config.results_dir / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            shutil.rmtree(latest_link)
        latest_link.symlink_to(self.config.benchmark_id)

        logger.info(f"Benchmark initialized: {self.config.benchmark_id}")
        logger.info(f"Output directory: {self.config.output_dir}")

    def load_queries(self) -> List[Dict[str, Any]]:
        """Load queries from query file."""
        if not self.config.query_file:
            raise ValueError("query_file is required in config")

        query_path = Path(self.config.query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")

        logger.info(f"Loading queries from: {query_path}")
        queries = load_queries_from_checkpoint(query_path)

        # Filter by configured levels
        if self.config.query_levels:
            queries = [
                q for q in queries
                if q.get("level", "") in self.config.query_levels
            ]

        logger.info(f"Loaded {len(queries)} queries")
        return queries

    async def generate_samples(
        self,
        queries: List[Dict[str, Any]],
    ) -> List[SampleResult]:
        """Generate code samples from all models."""
        all_results = []

        total_models = len(self.config.models)

        for model_idx, model in enumerate(self.config.models, 1):
            console.print(Panel(
                f"[bold cyan]Model {model_idx}/{total_models}:[/bold cyan] [yellow]{model}[/yellow]",
                title="[bold green]LLM Generation[/bold green]",
                border_style="green"
            ))

            results = await self.caller.generate_samples(
                model=model,
                queries=queries,
                num_samples=self.config.num_samples,
                save_dir=self.config.samples_dir if self.config.save_intermediate else None,
            )

            all_results.extend(results)
            console.print(f"  [green]✓[/green] Generated {len(results)} samples\n")

        return all_results

    def extract_codes(
        self,
        sample_results: List[SampleResult],
    ) -> List[Dict[str, Any]]:
        """Extract code from all sample results."""
        extracted = []

        for sample in sample_results:
            if not sample.success:
                extracted.append({
                    "query_id": sample.query_id,
                    "model": sample.model,
                    "sample_id": sample.sample_id,
                    "strategy_code": None,
                    "factor_codes": [],
                    "extraction_error": sample.error or "API call failed",
                })
                continue

            code = self.extractor.extract(sample.response)

            extracted.append({
                "query_id": sample.query_id,
                "model": sample.model,
                "sample_id": sample.sample_id,
                "strategy_code": code.strategy_code,
                "factor_codes": code.factor_codes,
                "extraction_method": code.extraction_method,
                "extraction_error": code.error,
            })

        # Save extracted codes
        if self.config.save_intermediate:
            codes_path = self.config.output_dir / "extracted_codes.json"
            with open(codes_path, "w", encoding="utf-8") as f:
                json.dump(extracted, f, indent=2, ensure_ascii=False)

        return extracted

    def run_backtests(
        self,
        extracted_codes: List[Dict[str, Any]],
    ) -> List[BacktestResult]:
        """Run backtests on all extracted codes."""
        logger.info(f"Running backtests on {len(extracted_codes)} samples...")

        results = self.runner.run_batch_backtest(
            extracted_codes,
            use_cache=True,
        )

        # Save backtest results
        if self.config.save_intermediate:
            results_path = self.config.output_dir / "backtest_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)

        return results

    def calculate_metrics(
        self,
        backtest_results: List[BacktestResult],
    ) -> Dict[str, Any]:
        """Calculate Pass@k metrics from backtest results."""
        all_metrics = self.metrics_calc.calculate_all_metrics(
            backtest_results,
            self.config.models,
        )

        # Print summary
        summary = self.metrics_calc.format_summary(all_metrics)
        print(summary)

        # Convert to JSON-serializable format
        metrics_dict = self.metrics_calc.to_json_dict(all_metrics)

        return metrics_dict

    def save_results(self, metrics: Optional[Dict[str, Any]]):
        """Save final results to disk."""
        # Build validation summary
        validation_summary = None
        if self.validation_result:
            validation_summary = {
                "computed_factors": self.validation_result.computed_factors,
                "invalid_factors_count": len(self.validation_result.invalid_factors),
                "affected_samples_count": len(self.validation_result.invalid_samples),
            }

        results = {
            "benchmark_id": self.config.benchmark_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "results": metrics,
            "summary": {
                "total_samples": len(self.sample_results),
                "total_extracted": len([c for c in self.extracted_codes if c.get("strategy_code")]),
                "total_passed": len([r for r in self.backtest_results if r.passed]) if self.backtest_results else 0,
                "backtest_skipped": metrics is None,
            },
            "validation": validation_summary,
        }

        results_path = self.config.output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {results_path}")

    async def run(self):
        """Run the full benchmark pipeline."""
        logger.info("=" * 60)
        logger.info("Starting LLM Benchmark")
        logger.info("=" * 60)

        # Setup
        self.setup()

        # Load queries
        queries = self.load_queries()

        # Generate samples
        logger.info("\n[Step 1/5] Generating code samples...")
        self.sample_results = await self.generate_samples(queries)
        logger.info(f"Generated {len(self.sample_results)} total samples")

        # Extract codes
        logger.info("\n[Step 2/5] Extracting code from responses...")
        self.extracted_codes = self.extract_codes(self.sample_results)
        valid_codes = len([c for c in self.extracted_codes if c.get("strategy_code")])
        logger.info(f"Extracted {valid_codes} valid strategy codes")

        # Validate and compute missing factors
        logger.info("\n[Step 3/5] Validating factors...")
        self.validation_result = await self.validator.validate_and_compute(
            self.extracted_codes
        )
        if self.validation_result.invalid_samples:
            logger.warning(
                f"Found {len(self.validation_result.invalid_samples)} samples "
                f"with invalid factors"
            )
            # Save invalid factors report
            report_path = self.config.output_dir / "invalid_factors.json"
            self.validator.save_invalid_report(self.validation_result, report_path)

        # Skip backtest if requested
        if self.skip_backtest:
            logger.info("\n[Step 4/5] Skipping backtests (--no-backtest)")
            logger.info("[Step 5/5] Skipping metrics calculation (--no-backtest)")

            # Save partial results
            self.save_results(metrics=None)

            logger.info("\n" + "=" * 60)
            logger.info("Code generation completed (backtest skipped)!")
            logger.info(f"Results: {self.config.output_dir}")
            logger.info("=" * 60)

            return None

        # Run backtests
        logger.info("\n[Step 4/5] Running backtests...")
        self.backtest_results = self.run_backtests(self.extracted_codes)
        passed = len([r for r in self.backtest_results if r.passed])
        logger.info(f"Passed {passed}/{len(self.backtest_results)} backtests")

        # Calculate metrics
        logger.info("\n[Step 5/5] Calculating metrics...")
        metrics = self.calculate_metrics(self.backtest_results)

        # Save results
        self.save_results(metrics)

        logger.info("\n" + "=" * 60)
        logger.info("Benchmark completed!")
        logger.info(f"Results: {self.config.output_dir / 'results.json'}")
        logger.info("=" * 60)

        return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Benchmark - Evaluate code generation quality using Pass@k"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config JSON file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models (overrides config file)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples per query (overrides config file)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config file)",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="Specific query file to use (overrides config file)",
    )
    parser.add_argument(
        "--benchmark-id",
        type=str,
        default=None,
        help="Custom benchmark ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent API requests (overrides config file)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of backtest results",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing benchmark: 'auto' to find compatible one, or specific benchmark_id",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest and metrics calculation after generating code samples (default: skip)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry samples that previously failed due to API errors (success=False)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict factor validation: use only base meta_info.json and treat unknown factors as "
             "errors. Default is permissive (compute missing factors via meta_info_auto.json), which "
             "reproduces the published results.",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Load config from file
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        config = BenchmarkConfig.from_file(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = BenchmarkConfig()

    # Override with command line arguments
    if args.models:
        config.models = args.models.split(",")
    if args.samples:
        config.num_samples = args.samples
    if args.temperature is not None:
        config.temperature = args.temperature
        # Also update the config file for future runs
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                config_data["temperature"] = args.temperature
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated {config_path} with temperature={args.temperature}")
            except Exception as e:
                logger.warning(f"Failed to update config file: {e}")
    if args.query_file:
        config.query_file = args.query_file
    if args.benchmark_id:
        config.benchmark_id = args.benchmark_id
    if args.max_concurrent:
        config.max_concurrent_requests = args.max_concurrent
    if args.no_cache:
        config.retry_failed = False

    # Run benchmark
    benchmark = LLMBenchmark(config, resume=args.resume, skip_backtest=not args.backtest, retry_failed=args.retry_failed, strict=args.strict)
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
