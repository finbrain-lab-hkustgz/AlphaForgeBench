"""Strategy Server

Server implementation for the Strategy Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
import os
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd

from src.logger import logger
from src.config import config
from src.strategy.context import StrategyContextManager
from src.strategy.types import Strategy, StrategyConfig
from src.utils import assemble_project_path
from src.dynamic import dynamic_manager

class StrategyManager(BaseModel):
    """Strategy Manager for managing strategy registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the strategies")
    save_path: str = Field(default=None, description="The path to save the strategies")
    contract_path: str = Field(default=None, description="The path to save the strategy contract")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the Strategy Manager."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, StrategyConfig] = {}  # strategy_name -> StrategyConfig

        
    async def initialize(self, strategy_names: Optional[List[str]] = None):
        """Initialize strategies by names using strategy context manager with concurrent support.
        
        Args:
            strategy_names: List of strategy names to initialize. If None, initialize all registered strategies.
        """
        
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "strategy"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "strategy.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Strategy Manager base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")
        
        # Initialize strategy context manager
        self.strategy_context_manager = StrategyContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
        )
        await self.strategy_context_manager.initialize(strategy_names=strategy_names)
        
        logger.info("| ✅ Strategies initialization completed")
        
    async def get_contract(self) -> str:
        """Get the contract for all strategies"""
        return await self.strategy_context_manager.load_contract()
    
    async def register(self, 
                       strategy: Union[str, Strategy, Type[Strategy]],
                       config: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> StrategyConfig:
        """Register a strategy from code string, class, or instance asynchronously.
        
        Args:
            strategy: Strategy code string, Strategy class, or Strategy instance to register
            config: Configuration dict for strategy initialization (required when strategy is a class or code)
            override: Whether to override existing registration
            version: Optional version string
            
        Returns:
            StrategyConfig: Strategy configuration
            
        Raises:
            ValueError: If strategy code cannot be loaded or strategy type is invalid
        """
        # Handle code string - load strategy class dynamically
        if isinstance(strategy, str):
            logger.info(f"| 📝 Loading strategy from code...")
            strategy_cls = await self._load_strategy_from_code(strategy)
            # Use empty config if not provided
            if config is None:
                config = {}
            # Pass the code string to context manager
            strategy_config = await self.strategy_context_manager.register(
                strategy_cls, 
                strategy_config_dict=config, 
                override=override,
                version=version,
                code=strategy  # Pass the original code string
            )
        # Handle Strategy instance
        elif isinstance(strategy, Strategy):
            strategy_cls = type(strategy)
            # Use instance's config if config not provided
            if config is None:
                config = {}
            strategy_config = await self.strategy_context_manager.register(
                strategy_cls, 
                strategy_config_dict=config, 
                override=override,
                version=version
            )
        # Handle Strategy class
        else:
            strategy_cls = strategy
            strategy_config = await self.strategy_context_manager.register(
                strategy_cls, 
                strategy_config_dict=config, 
                override=override,
                version=version
            )
        self._registered_configs[strategy_config.name] = strategy_config
        logger.info(f"| ✅ Strategy registered: {strategy_config.name}")
        return strategy_config
    
    async def _load_strategy_from_code(self, code: str) -> Type[Strategy]:
        """
        Load strategy class from code string using dynamic_manager.
        
        This follows the same pattern as StrategyConfig.model_validate.
        
        Args:
            code: Strategy code string
            
        Returns:
            Strategy class
            
        Raises:
            ValueError: If class name cannot be extracted or class cannot be loaded
        """
        # Handle escaped characters (e.g., \\n -> \n)
        # This can happen when code is serialized/deserialized through JSON
        import codecs
        try:
            # Try to decode escaped strings (e.g., \\n -> \n)
            decoded_code = codecs.decode(code, 'unicode_escape')
            # Only use decoded code if it's different and valid
            if decoded_code != code and len(decoded_code) > 0:
                code = decoded_code
        except Exception:
            # If decoding fails, use original code
            pass
        
        # Try to extract class name first
        class_name = dynamic_manager.extract_class_name_from_code(code)
        
        # Load the class using dynamic_manager
        # If class_name extraction failed, load_class will try to find it automatically using base_class
        try:
            if class_name:
                strategy_cls = dynamic_manager.load_class(
                    code,
                    class_name=class_name,
                    base_class=Strategy,
                    context="strategy"
                )
            else:
                # Fallback: let load_class find the class automatically using base_class
                strategy_cls = dynamic_manager.load_class(
                    code,
                    class_name=None,
                    base_class=Strategy,
                    context="strategy"
                )
        except Exception as e:
            raise ValueError(f"Failed to load strategy class from code: {e}")
        
        if strategy_cls is None:
            raise ValueError(f"Failed to load strategy class from code")
        
        return strategy_cls
    
    async def list(self) -> List[str]:
        """List all registered strategies
        
        Returns:
            List[str]: List of strategy names
        """
        return await self.strategy_context_manager.list()
    
    
    async def get(self, strategy_name: str) -> Strategy:
        """Get strategy instance by name
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Strategy: Strategy instance or None if not found
        """
        strategy = await self.strategy_context_manager.get(strategy_name)
        return strategy
    
    async def get_info(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get strategy configuration by name
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            StrategyConfig: Strategy configuration or None if not found
        """
        return await self.strategy_context_manager.get_info(strategy_name)
    
    async def cleanup(self):
        """Cleanup all strategies"""
        await self.strategy_context_manager.cleanup()
        self._registered_configs.clear()
    
    async def update(self, 
                     strategy: Union[str, Strategy, Type[Strategy]], 
                     config: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> StrategyConfig:
        """Update an existing strategy with new configuration and create a new version
        
        Args:
            strategy: New strategy code string, class, or instance with updated implementation
            config: Configuration dict for strategy initialization (required when strategy is a class or code)
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            StrategyConfig: Updated strategy configuration
            
        Raises:
            ValueError: If strategy code cannot be loaded or strategy type is invalid
        """
        # Handle code string - load strategy class dynamically
        if isinstance(strategy, str):
            logger.info(f"| 📝 Loading strategy from code for update...")
            strategy_cls = await self._load_strategy_from_code(strategy)
            # Use empty config if not provided
            if config is None:
                config = {}
            # Pass the code string to context manager
            strategy_config = await self.strategy_context_manager.update(
                strategy_cls, 
                strategy_config_dict=config, 
                new_version=new_version, 
                description=description,
                code=strategy  # Pass the original code string
            )
        # Handle Strategy instance
        elif isinstance(strategy, Strategy):
            strategy_cls = type(strategy)
            # Use instance's config if config not provided
            if config is None:
                config = {}
            strategy_config = await self.strategy_context_manager.update(
                strategy_cls, 
                strategy_config_dict=config, 
                new_version=new_version, 
                description=description
            )
        # Handle Strategy class
        else:
            strategy_cls = strategy
            strategy_config = await self.strategy_context_manager.update(
                strategy_cls, 
                strategy_config_dict=config, 
                new_version=new_version, 
                description=description
            )
        self._registered_configs[strategy_config.name] = strategy_config
        logger.info(f"| ✅ Strategy updated: {strategy_config.name}")
        return strategy_config
    
    async def copy(self, strategy_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, new_config: Optional[Dict[str, Any]] = None) -> StrategyConfig:
        """Copy an existing strategy
        
        Args:
            strategy_name: Name of the strategy to copy
            new_name: New name for the copied strategy. If None, uses original name.
            new_version: New version for the copied strategy. If None, increments version.
            new_config: New configuration dict for the copied strategy. If None, uses original config.
            
        Returns:
            StrategyConfig: New strategy configuration
        """
        strategy_config = await self.strategy_context_manager.copy(
            strategy_name, new_name, new_version, new_config
        )
        self._registered_configs[strategy_config.name] = strategy_config
        return strategy_config
    
    async def unregister(self, strategy_name: str) -> bool:
        """Unregister a strategy
        
        Args:
            strategy_name: Name of the strategy to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.strategy_context_manager.unregister(strategy_name)
        if success and strategy_name in self._registered_configs:
            del self._registered_configs[strategy_name]
        return success
    
    async def restore(self, strategy_name: str, version: str, auto_initialize: bool = True) -> Optional[StrategyConfig]:
        """Restore a specific version of a strategy from history
        
        Args:
            strategy_name: Name of the strategy
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored strategy
            
        Returns:
            StrategyConfig of the restored version, or None if not found
        """
        strategy_config = await self.strategy_context_manager.restore(strategy_name, version, auto_initialize)
        if strategy_config:
            self._registered_configs[strategy_config.name] = strategy_config
        return strategy_config
    
    async def __call__(self, strategy_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Call a strategy by name
        
        Args:
            strategy_name: Strategy name
            df: Input DataFrame containing price and factor data
            
        Returns:
            Dict[str, Any]: Strategy output containing signal and position
        """
        strategy = await self.get(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy {strategy_name} not found")
        return await strategy(df)


# Global Strategy manager instance
strategy_manager = StrategyManager()

