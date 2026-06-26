workdir = "workdir"
assets_name = "crypto"
source = "binance"
data_type = "price"
level = "1day"
tag = f"{assets_name}_{source}_{data_type}_{level}"
workdir = f"{workdir}/{tag}"
log_path = "fs.log"

downloader = dict(
    type = "PriceDownloader",
    source = source,
    assets_path = f"configs/_asset_list_/{assets_name}.json",
    # start_date = "2017-09-01",
    start_date = "2025-01-01",
    end_date = "2026-01-01",
    level=level,
    format="%Y-%m-%d %H:%M:%S",
    max_concurrent = 3,
)