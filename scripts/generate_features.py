#!/usr/bin/env python3
"""
从 datasets/market/market_price_1day 生成 feature 数据
输出到 datasets/market/market_feature_1day
"""
import os
import sys
import asyncio
import pandas as pd

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factor import factor_manager
from src.utils import gather_with_concurrency

# 配置
PRICE_DIR = "datasets/market/market_price_1day"
FEATURE_DIR = "datasets/market/market_feature_1day"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
MAX_CONCURRENT = 6

# 因子列表
FACTORS = [
    'atr', 'bb', 'beta', 'cci', 'cnt', 'cord', 'corr', 'ema',
    'imax', 'imin', 'imxd', 'kdj', 'klen', 'klow', 'kmid', 'kup',
    'ksft', 'logvol', 'ma', 'macd', 'max', 'mfi', 'min', 'obv',
    'qtld', 'qtlu', 'rank', 'roc', 'rsi', 'rsv', 'sma', 'std',
    'sum', 'vsum', 'vma', 'vstd', 'wvma',
]

async def process_symbol(symbol: str):
    """处理单个交易对"""
    print(f"Processing {symbol}...")

    price_path = os.path.join(PRICE_DIR, f"{symbol}.jsonl")
    feature_path = os.path.join(FEATURE_DIR, f"{symbol}.jsonl")

    # 读取价格数据
    df = pd.read_json(price_path, lines=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp")
    df = df.reset_index(drop=True)

    # 保存 timestamp 列
    timestamp_col = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"  Loaded {len(df)} rows, computing {len(FACTORS)} factors...")

    # 计算所有因子
    factors_df = await gather_with_concurrency(
        [factor_manager(factor_name, df) for factor_name in FACTORS],
        max_concurrency=MAX_CONCURRENT,
    )
    factors_df = pd.concat(factors_df, axis=1)

    # 添加 timestamp 列到第一列
    factors_df["timestamp"] = timestamp_col.values
    columns = factors_df.columns.tolist()
    columns.remove("timestamp")
    columns.insert(0, "timestamp")
    factors_df = factors_df[columns]

    # 保存
    factors_df.to_json(feature_path, orient="records", lines=True, index=False)

    print(f"  Saved {len(factors_df)} rows to {feature_path}")
    print(f"  Date range: {timestamp_col.iloc[0]} to {timestamp_col.iloc[-1]}")

    return {
        "symbol": symbol,
        "rows": len(factors_df),
        "start_date": timestamp_col.iloc[0],
        "end_date": timestamp_col.iloc[-1],
    }

async def main():
    print("=" * 60)
    print("Generating market feature data")
    print("=" * 60)
    print(f"Price dir: {PRICE_DIR}")
    print(f"Feature dir: {FEATURE_DIR}")
    print(f"Symbols: {SYMBOLS}")
    print("=" * 60)

    # 初始化 factor_manager
    print("Initializing factor manager...")
    await factor_manager.initialize(factor_names=FACTORS)
    print("Factor manager initialized.")

    os.makedirs(FEATURE_DIR, exist_ok=True)

    results = []
    for symbol in SYMBOLS:
        result = await process_symbol(symbol)
        results.append(result)

    print("\n" + "=" * 60)
    print("Summary:")
    for r in results:
        print(f"  {r['symbol']}: {r['rows']} rows ({r['start_date']} ~ {r['end_date']})")
    print("=" * 60)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
