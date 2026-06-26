import argparse
import os
import sys
from pathlib import Path
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from src.logger import logger
from src.config import config
from src.registry import DOWNLOADER

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "download", "crypto", "crypto_binance_price_1min.py"), help="config file path")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

async def main():
    args = parse_args()
    
    config.initialize(config_path = args.config, args = args)
    logger.initialize(config = config)
    logger.info(f"| Config: {config.pretty_text}")

    downloader = DOWNLOADER.build(config.downloader)

    try:
        await downloader.run()
    except KeyboardInterrupt:
        sys.exit()


if __name__ == '__main__':
    asyncio.run(main())