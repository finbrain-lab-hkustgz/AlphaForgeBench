#!/usr/bin/env python3
"""Download the AlphaForgeBench dataset (and the example benchmark answers) from the
Hugging Face Hub into this repository.

The large data artifacts are hosted on the Hub rather than committed to git:
  - datasets/market/                                                (price + factor-feature data, ~140 MB)
  - AlphaForgeBench/benchmark_results/bench_t={0,0.7}/extracted_codes.json   (model answers, ~52 MB each)
  - AlphaForgeBench/benchmark_results/bench_t={0,0.7}/query_metrics.json     (per-query metrics, ~7 MB each)

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --repo-id <org>/<dataset-name>
    HF_DATA_REPO=<org>/<dataset-name> python scripts/download_data.py

The default repo id can also be overridden with the HF_DATA_REPO environment variable.
A Hugging Face token is only required if the dataset repo is private
(set HF_API_KEY / HF_TOKEN, or run `huggingface-cli login`).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Default Hugging Face dataset repo holding the data. Override with --repo-id or HF_DATA_REPO.
DEFAULT_REPO_ID = "finbrain-lab-hkustgz/AlphaForgeBench-data"

# Repo root = parent of this scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AlphaForgeBench data from the Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_DATA_REPO", DEFAULT_REPO_ID),
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dest",
        default=str(REPO_ROOT),
        help="Destination directory (default: repository root)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required. Install it with: pip install -r requirements.txt"
        ) from exc

    token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")

    print(f"Downloading '{args.repo_id}' into {args.dest} ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.dest,
        token=token,
        # Only the data trees — NOT the Hub-side README.md (dataset card), which would
        # otherwise overwrite this repository's own README.
        allow_patterns=["datasets/**", "AlphaForgeBench/**"],
    )
    print("Done. Expected layout:")
    print("  datasets/market/{market_price_1day, market_feature_1day, factor, meta_info*.json}")
    print("  AlphaForgeBench/benchmark_results/bench_t={0,0.7}/{extracted_codes.json, query_metrics.json}")


if __name__ == "__main__":
    main()
