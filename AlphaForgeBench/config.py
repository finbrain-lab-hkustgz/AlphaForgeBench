"""Configuration for LLM benchmark evaluation."""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


# Default config file path
DEFAULT_CONFIG_PATH = Path("AlphaForgeBench/configs/benchmark_config.json")

# Query levels (matching generated_queries.json format)
QUERY_LEVELS = [
    "L1_easy",
    "L1_medium",
    "L1_hard",
    "L2_easy",
    "L2_medium",
    "L2_hard",
    "L3_easy",
    "L3_medium",
    "L3_hard",
]

def load_config_from_file(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    # Models to evaluate
    models: List[str] = field(default_factory=list)

    # Sampling parameters
    num_samples: int = 5  # k in Pass@k
    temperature: float = 0.8

    # Query configuration
    query_levels: List[str] = field(default_factory=lambda: QUERY_LEVELS.copy())
    query_file: str = ""  # Path to query file (e.g., AlphaForgeBench/generated_queries/xxx/generated_queries.json)

    # Output configuration
    results_dir: Path = Path("AlphaForgeBench/benchmark_results")
    benchmark_id: Optional[str] = None  # Auto-generated if not provided

    # System prompt
    system_prompt_path: Path = Path("AlphaForgeBench/prompts/system_prompt.txt")
    system_prompt: Optional[str] = None  # Loaded from file if not provided

    # Backtest configuration
    data_config: dict = field(default_factory=lambda: {
        "asset_name": "market",
        "data_type": "price",
        "level": "1day",
    })
    start_ts: str = "2024-01-01 00:00:00"
    end_ts: str = "2024-12-31 00:00:00"
    history_ts: int = 120
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    data_path: str = "datasets/market"

    # Execution options
    max_concurrent_requests: int = 5
    retry_failed: bool = True
    save_intermediate: bool = True

    # Source config file path (for copying to results)
    _source_config_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize derived fields."""
        # Generate benchmark ID if not provided
        if self.benchmark_id is None:
            self.benchmark_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Convert paths to Path objects
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        if isinstance(self.system_prompt_path, str):
            self.system_prompt_path = Path(self.system_prompt_path)

        # Load system prompt if not provided
        if self.system_prompt is None and self.system_prompt_path.exists():
            self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8")

    @classmethod
    def from_file(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> "BenchmarkConfig":
        """Create config from JSON file."""
        data = load_config_from_file(config_path)

        # Extract backtest config if nested
        backtest = data.pop("backtest", {})

        config = cls(
            models=data.get("models", []),
            num_samples=data.get("num_samples", 5),
            temperature=data.get("temperature", 0.8),
            query_levels=data.get("query_levels", QUERY_LEVELS.copy()),
            query_file=data.get("query_file", ""),
            results_dir=Path(data.get("results_dir", "AlphaForgeBench/benchmark_results")),
            system_prompt_path=Path(data.get("system_prompt_path", "AlphaForgeBench/prompts/system_prompt.txt")),
            # Backtest config
            data_config=backtest.get("data_config", data.get("data_config", {})),
            start_ts=backtest.get("start_ts", data.get("start_ts", "2024-01-01 00:00:00")),
            end_ts=backtest.get("end_ts", data.get("end_ts", "2024-12-31 00:00:00")),
            history_ts=backtest.get("history_ts", data.get("history_ts", 120)),
            symbols=backtest.get("symbols", data.get("symbols", ["BTCUSDT", "ETHUSDT"])),
            data_path=backtest.get("data_path", data.get("data_path", "datasets/market")),
            # Execution options
            max_concurrent_requests=data.get("max_concurrent_requests", 5),
            save_intermediate=data.get("save_intermediate", True),
        )

        # Store source config path for later copying
        config._source_config_path = config_path

        return config

    @property
    def output_dir(self) -> Path:
        """Get the output directory for this benchmark run."""
        return self.results_dir / self.benchmark_id

    @property
    def samples_dir(self) -> Path:
        """Get the directory for storing generated samples."""
        return self.output_dir / "samples"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory for backtest results."""
        return self.output_dir / "cache"

    def copy_config_to_output(self):
        """Copy the source config file to the output directory."""
        if self._source_config_path and self._source_config_path.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            dest_path = self.output_dir / "benchmark_config.json"
            shutil.copy2(self._source_config_path, dest_path)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "models": self.models,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "query_levels": self.query_levels,
            "query_file": self.query_file,
            "results_dir": str(self.results_dir),
            "benchmark_id": self.benchmark_id,
            "system_prompt_path": str(self.system_prompt_path),
            "backtest": {
                "data_config": self.data_config,
                "start_ts": self.start_ts,
                "end_ts": self.end_ts,
                "history_ts": self.history_ts,
                "symbols": self.symbols,
                "data_path": self.data_path,
            },
            "max_concurrent_requests": self.max_concurrent_requests,
            "retry_failed": self.retry_failed,
            "save_intermediate": self.save_intermediate,
        }

    def get_compatible_config_dict(self) -> dict:
        """
        Get config dict for compatibility comparison.

        Only includes fields that must match for two benchmark runs to be compatible.
        Excludes: models, benchmark_id, num_samples, max_concurrent_requests, save_intermediate, retry_failed

        Note: num_samples is excluded to allow incremental sample generation
        (e.g., run with num_samples=1 first, then resume with num_samples=3)
        """
        return {
            "temperature": self.temperature,
            "query_levels": sorted(self.query_levels),
            "query_file": self.query_file,
            "system_prompt_path": str(self.system_prompt_path),
            "backtest": {
                "data_config": self.data_config,
                "start_ts": self.start_ts,
                "end_ts": self.end_ts,
                "history_ts": self.history_ts,
                "symbols": sorted(self.symbols),
                "data_path": self.data_path,
            },
        }

    def is_compatible_with(self, other: "BenchmarkConfig") -> bool:
        """
        Check if this config is compatible with another config.

        Two configs are compatible if they can share results (same sampling params,
        query config, and backtest config). Models can differ.
        """
        return self.get_compatible_config_dict() == other.get_compatible_config_dict()

    @classmethod
    def from_saved_config(cls, config_path: Path) -> "BenchmarkConfig":
        """Load config from a saved benchmark_config.json file."""
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        backtest = data.get("backtest", {})

        config = cls(
            models=data.get("models", []),
            num_samples=data.get("num_samples", 5),
            temperature=data.get("temperature", 0.8),
            query_levels=data.get("query_levels", QUERY_LEVELS.copy()),
            query_file=data.get("query_file", ""),
            results_dir=Path(data.get("results_dir", "AlphaForgeBench/benchmark_results")),
            benchmark_id=data.get("benchmark_id"),
            system_prompt_path=Path(data.get("system_prompt_path", "AlphaForgeBench/prompts/system_prompt.txt")),
            # Backtest config
            data_config=backtest.get("data_config", {}),
            start_ts=backtest.get("start_ts", "2024-01-01 00:00:00"),
            end_ts=backtest.get("end_ts", "2024-12-31 00:00:00"),
            history_ts=backtest.get("history_ts", 120),
            symbols=backtest.get("symbols", ["BTCUSDT", "ETHUSDT"]),
            data_path=backtest.get("data_path", "datasets/market"),
            # Execution options
            max_concurrent_requests=data.get("max_concurrent_requests", 5),
            save_intermediate=data.get("save_intermediate", True),
            retry_failed=data.get("retry_failed", True),
        )

        return config
