import os
import sys
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.version import version_manager
from src.factor import factor_manager
from src.registry import PROCESSOR

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "process", "crypto.py"), help="config file path")

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
    
    # Initialize factor manager
    logger.info("| 🧠 Initializing factor manager...")
    await factor_manager.initialize(factor_names = config.factors if "factors" in config else None)
    logger.info(f"| ✅ Factor manager initialized: {await factor_manager.list()}")
    
    # Initialize version manager
    logger.info("| 🧠 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {await version_manager.list()}")

    processor = PROCESSOR.build(config.processor)

    try:
        await processor.run()
    except KeyboardInterrupt:
        sys.exit()


if __name__ == '__main__':
    asyncio.run(main())