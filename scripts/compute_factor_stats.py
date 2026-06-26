"""
Compute factor statistics from BTCUSDT data.
Outputs min, Q1, median, Q3, max for each factor.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np


# Define the fixed factor list (126 factors from meta_info.json)
FIXED_FACTORS = [
    # Technical indicators (fixed periods)
    "atr_14", "bb_lower_20", "bb_middle_20", "bb_upper_20",
    "cci_14", "ema_20", "ema_50", "macd", "macd_hist", "macd_signal",
    "mfi_14", "obv", "rsi_14", "sma_20", "sma_50", "stoch_d_14", "stoch_k_14",

    # Statistical factors (periods: 5, 10, 20, 30, 60)
    *[f"beta_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"corr_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"cord_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"std_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"vstd_{p}" for p in [5, 10, 20, 30, 60]],

    # Time series factors (periods: 5, 10, 20, 30, 60)
    *[f"imax_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"imin_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"imxd_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"ma_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"max_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"min_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"qtld_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"qtlu_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"rank_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"roc_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"rsv_{p}" for p in [5, 10, 20, 30, 60]],

    # Candlestick pattern factors (no period)
    "klen", "klow", "klow2", "kmid", "kmid2", "ksft", "ksft2", "kup", "kup2",

    # Volume factors
    "logvol",
    *[f"vma_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"wvma_{p}" for p in [5, 10, 20, 30, 60]],

    # Counting factors (periods: 5, 10, 20, 30, 60)
    *[f"cntp_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"cntn_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"cntd_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"sump_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"sumn_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"sumd_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"vsump_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"vsumn_{p}" for p in [5, 10, 20, 30, 60]],
    *[f"vsumd_{p}" for p in [5, 10, 20, 30, 60]],
]


def load_btcusdt_data() -> pd.DataFrame:
    """Load BTCUSDT feature data from JSONL file."""
    data_path = Path("datasets/market/market_feature_1day/BTCUSDT.jsonl")

    records = []
    with open(data_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    return df


def compute_stats(df: pd.DataFrame, factors: list[str]) -> dict:
    """Compute statistics for each factor."""
    stats = {}

    for factor in factors:
        if factor not in df.columns:
            print(f"Warning: Factor '{factor}' not found in data")
            continue

        series = df[factor].dropna()

        if len(series) == 0:
            print(f"Warning: Factor '{factor}' has no valid data")
            continue

        stats[factor] = {
            "min": float(series.min()),
            "Q1": float(series.quantile(0.25)),
            "median": float(series.quantile(0.50)),
            "Q3": float(series.quantile(0.75)),
            "max": float(series.max()),
            "count": int(len(series)),
        }

    return stats


def main():
    print("Loading BTCUSDT data...")
    df = load_btcusdt_data()
    print(f"Loaded {len(df)} rows")

    print(f"\nComputing statistics for {len(FIXED_FACTORS)} factors...")
    stats = compute_stats(df, FIXED_FACTORS)

    # Save to outputs/factor_stats.json
    output_path = Path("outputs/factor_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved statistics to {output_path}")
    print(f"Total factors with stats: {len(stats)}")

    # Print summary
    print("\n=== Sample Statistics ===")
    sample_factors = ["rsi_14", "macd", "std_20", "rank_20", "cntp_20"]
    for factor in sample_factors:
        if factor in stats:
            s = stats[factor]
            print(f"{factor}: [{s['min']:.4f}, {s['Q1']:.4f}, {s['median']:.4f}, {s['Q3']:.4f}, {s['max']:.4f}]")


if __name__ == "__main__":
    main()
