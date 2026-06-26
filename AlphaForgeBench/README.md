# AlphaForgeBench (package)

This package implements the benchmark. For installation, data download, and the full
walkthrough, see the [repository README](../README.md). All commands are run from the
**repository root**.

## Modules

| Module | Role |
|---|---|
| `benchmark.py` | End-to-end driver: sample → extract → validate → backtest → metrics. CLI: `python -m AlphaForgeBench.benchmark` |
| `query_batch_generator.py` | Stage 1 — generate benchmark queries with an LLM (reads `configs/generation_config.json`) |
| `api_caller.py` | Stage 2 — async, cached sampling of model answers |
| `code_extractor.py` | Parse strategy/factor code out of model responses (JSON / code-block / raw-class) |
| `factor_validator.py` | Detect missing/invalid factors; optionally compute them against the dataset |
| `backtest_runner.py` | Sandboxed exec of generated code + per-symbol/yearly backtest + metric aggregation |
| `metrics.py` | Pass@k, Pass@1, syntax-pass-rate, and averaged financial metrics |
| `run_backtest.py` | Standalone re-backtest of an existing `samples/` directory |
| `config.py` | `BenchmarkConfig` (load/save, resume/compatibility) |

## Data contract

The backtest reads `datasets/market/` (downloaded from the Hub — see the root README):

```
datasets/market/
├── market_price_1day/      # <SYMBOL>.jsonl daily OHLCV
├── market_feature_1day/    # <SYMBOL>.jsonl daily factor features
├── factor/                 # factor.json + contract
├── meta_info.json          # tags = [market_price_1day, market_feature_1day], symbols, factor names
└── meta_info_auto.json
```

The dataset tag is built as `<asset_name>_<data_type>_<level>` (e.g. `market_price_1day`).
`benchmark_config.json` therefore sets `data_config = {asset_name: "market", data_type: "price", level: "1day"}`
and `data_path = "datasets/market"`.

## CLI reference

**`python -m AlphaForgeBench.benchmark`**

| Flag | Meaning |
|---|---|
| `--config PATH` | config file (default `AlphaForgeBench/configs/benchmark_config.json`) |
| `--models "a,b"` | comma-separated OpenRouter model ids (overrides config) |
| `--samples N` | samples per query (`k` in Pass@k) |
| `--temperature T` | sampling temperature |
| `--backtest` | enable backtest + metrics (otherwise stops after extraction) |
| `--strict` | strict factor validation (base `meta_info.json` only); **default is permissive** — compute missing factors via `meta_info_auto.json`, matching the published results |
| `--resume auto` | resume a compatible previous run |
| `--retry-failed` | retry samples whose API call previously failed |
| `--no-cache` | disable the backtest cache |
| `--query-file PATH` | override the query file |
| `--max-concurrent N` | max concurrent API requests |

**`python -m AlphaForgeBench.run_backtest`**

| Flag | Meaning |
|---|---|
| `--extracted-codes PATH` | re-backtest a published `extracted_codes.json` directly (skips extraction) — reproduces the shipped scores |
| `--samples-dir PATH` | re-score a directory of raw per-sample files instead (exactly one of the two is required) |
| `--workers N` | parallel workers (default 1) |
| `--pass-count N` | only score samples with `sample_id < N` (e.g. Pass@1) |
| `--symbol SYM` | restrict to one symbol |
| `--allow-compute-factors` | compute factors missing from `meta_info.json` on the fly |
| `--watch` / `--watch-timeout S` | keep processing newly added samples |
| `--clear-cache` / `--retry-errors` / `--retry-keys` | cache management |
