import os
from mmengine import Config as MMConfig
from argparse import Namespace
from typing import Union

from src.utils import assemble_project_path, Singleton

def process_general(config: MMConfig) -> MMConfig:
    """Process general configuration and ensure paths are strings"""
    workdir = str(assemble_project_path(config.workdir))
    os.makedirs(workdir, exist_ok=True)
    config.workdir = workdir
    
    log_path = getattr(config, 'log_path', 'agent.log')
    log_path = str(assemble_project_path(os.path.join(workdir, log_path)))
    config.log_path = log_path

    return config

class Config(MMConfig, metaclass=Singleton):
    def __init__(self):
        super(Config, self).__init__()

    def initialize(self, config_path: Union[str], args: Namespace) -> None:
        # Initialize the general configuration
        config_path = str(assemble_project_path(config_path))
        mmconfig = MMConfig.fromfile(filename=config_path)
        if 'cfg_options' not in args or args.cfg_options is None:
            cfg_options = dict()
        else:
            cfg_options = args.cfg_options
        for item in args.__dict__:
            if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
                cfg_options[item] = args.__dict__[item]
        mmconfig.merge_from_dict(cfg_options)

        # Process general configuration
        mmconfig = process_general(mmconfig)

        self.__dict__.update(mmconfig.__dict__)
    
    def dump(self) -> str:
        """Dump the configuration"""
        return super().dump()

config = Config()
config.initialize(config_path="configs/base.py", args=Namespace())