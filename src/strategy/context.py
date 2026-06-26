"""Strategy Context Manager for managing strategy lifecycle and resources with lazy loading."""
import os
import importlib
import inspect
import pkgutil
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union, Tuple
from datetime import datetime
import inflection
import json
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.utils import (assemble_project_path, 
                       gather_with_concurrency,
                       file_lock
                       )
from src.strategy.types import Strategy, StrategyConfig
from src.version import version_manager
from src.dynamic import dynamic_manager

class StrategyContextManager(BaseModel):
    """Global context manager for all strategies with lazy loading support."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the strategies")
    save_path: str = Field(default=None, description="The path to save the strategies")
    contract_path: str = Field(default=None, description="The path to save the strategy contract")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 contract_path: Optional[str] = None,
                 **kwargs):
        """Initialize the strategy context manager.
        
        Args:
            base_dir: Base directory for storing strategy data
            save_path: Path to save strategy configurations
            contract_path: Path to save strategy contract
        """
        super().__init__(**kwargs)
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "strategy"))
        logger.info(f"| 📁 Strategy context manager base directory: {self.base_dir}.")    
        os.makedirs(self.base_dir, exist_ok=True)
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "strategy.json")
        logger.info(f"| 📁 Strategy context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Strategy context manager contract path: {self.contract_path}.")

        self._strategy_configs: Dict[str, StrategyConfig] = {}  # Current active configs (latest version)
        # Strategy version history, e.g., {"strategy_name": {"1.0.0": StrategyConfig, "1.0.1": StrategyConfig}}
        self._strategy_history_versions: Dict[str, Dict[str, StrategyConfig]] = {}
        
        self._cleanup_registered = False
        
    async def initialize(self, strategy_names: Optional[List[str]] = None):
        """Initialize the strategy context manager."""
        
        # Register strategy-related symbols for auto-injection in dynamic code
        dynamic_manager.register_symbol("Strategy", Strategy)
        dynamic_manager.register_symbol("StrategyConfig", StrategyConfig)
        
        # Register strategy context provider for automatic import injection
        def strategy_context_provider():
            """Provide strategy-related imports for dynamic strategy classes."""
            return {
                "Strategy": Strategy,
                "StrategyConfig": StrategyConfig,
            }
        dynamic_manager.register_context_provider("strategy", strategy_context_provider)
        
        # Load strategies from module
        module_strategy_configs: Dict[str, StrategyConfig] = await self._load_from_module()
        
        # Load strategies from code (JSON file)
        code_strategy_configs: Dict[str, StrategyConfig] = await self._load_from_code()
        
        # Merge code configs with module configs, only override if code version is strictly greater
        strategy_configs = {}
        strategy_configs.update(module_strategy_configs)
        
        for strategy_name, code_config in code_strategy_configs.items():
            if strategy_name in strategy_configs:
                module_config = strategy_configs[strategy_name]
                # Compare versions: only override if code version is strictly greater
                if version_manager.compare_versions(code_config.version, module_config.version) > 0:
                    logger.info(f"| 🔄 Overriding strategy {strategy_name} from module (v{module_config.version}) with code version (v{code_config.version})")
                    strategy_configs[strategy_name] = code_config
                else:
                    logger.info(f"| 📌 Keeping strategy {strategy_name} from module (v{module_config.version}), code version (v{code_config.version}) is not greater")
                    # If versions are equal, update the history with module config (which has real class, not dynamic)
                    if version_manager.compare_versions(code_config.version, module_config.version) == 0:
                        # Replace the code config in history with module config to preserve real class reference
                        if strategy_name in self._strategy_history_versions:
                            self._strategy_history_versions[strategy_name][module_config.version] = module_config
            else:
                # New strategy from code, add it
                strategy_configs[strategy_name] = code_config
        
        # Filter strategies by names if provided
        if strategy_names is not None:
            strategy_configs = {name: strategy_configs[name] for name in strategy_names if name in strategy_configs}
        
        # Build all strategies concurrently with a concurrency limit
        strategy_names_list = list(strategy_configs.keys())
        tasks = [
            self.build(strategy_configs[name]) for name in strategy_names_list
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for strategy_name, result in zip(strategy_names_list, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize strategy {strategy_name}: {result}")
                continue
            self._strategy_configs[strategy_name] = result
            logger.info(f"| 🔧 Strategy {strategy_name} initialized")
        
        # Save strategy configs to json file
        await self.save_to_json()
        # Save contract to file (only for successfully initialized strategies)
        await self.save_contract()
        
        # Register cleanup callback
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True
        
        logger.info(f"| ✅ Strategies initialization completed")
        
    async def _load_from_module(self):
        """Load strategies from module (strategy subdirectories).
        
        Automatically discovers all Strategy classes from src/strategy subdirectories:
        - src/strategy/single_trading/
        - src/strategy/multi_trading/
        - src/strategy/portfolio/
        
        Returns:
            Dict[str, StrategyConfig]: Dictionary mapping strategy names to their configs
        """
        strategy_configs: Dict[str, StrategyConfig] = {}
        
        # Discover all Strategy classes from strategy subdirectories
        strategy_classes = []
        
        # Import the strategy package
        try:
            strategy_package = importlib.import_module("src.strategy")
        except ImportError as e:
            logger.error(f"| ❌ Failed to import strategy package: {e}")
            return strategy_configs
        
        # Strategy subdirectories to scan
        strategy_subdirs = ["single_trading", "multi_trading", "portfolio", "agent", "futures"]
        
        # Iterate through all subdirectories
        for subdir in strategy_subdirs:
            try:
                subdir_module = importlib.import_module(f"src.strategy.{subdir}")
                if not hasattr(subdir_module, "__path__"):
                    continue
                
                # Iterate through all modules in the subdirectory
                for importer, modname, ispkg in pkgutil.iter_modules(subdir_module.__path__, subdir_module.__name__ + "."):
                    if ispkg:
                        continue  # Skip sub-packages
                    
                    try:
                        # Import the module
                        module = importlib.import_module(modname)
                        
                        # Find all Strategy subclasses in the module
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, Strategy) and 
                                obj is not Strategy and 
                                obj.__module__ == modname):
                                strategy_classes.append(obj)
                                logger.debug(f"| 🔍 Discovered strategy class: {name} from {modname}")
                    except Exception as e:
                        logger.warning(f"| ⚠️ Failed to import module {modname}: {e}")
                        continue
            except ImportError as e:
                logger.warning(f"| ⚠️ Failed to import strategy subdirectory {subdir}: {e}")
                continue
        
        async def register_strategy_class(strategy_cls: Type[Strategy]):
            """Register a strategy class.
            
            Args:
                strategy_cls: Strategy class to register
            """
            try:
                # Get strategy properties from strategy class
                strategy_name_field = strategy_cls.model_fields.get('name', None)
                if strategy_name_field and hasattr(strategy_name_field, 'default'):
                    strategy_name_value = strategy_name_field.default
                else:
                    strategy_name_value = inflection.underscore(strategy_cls.__name__)
                
                strategy_description_field = strategy_cls.model_fields.get('description', None)
                if strategy_description_field and hasattr(strategy_description_field, 'default'):
                    strategy_description_value = strategy_description_field.default
                else:
                    strategy_description_value = f"{strategy_cls.__name__} strategy"
                
                strategy_factor_names_field = strategy_cls.model_fields.get('factor_names', None)
                if strategy_factor_names_field and hasattr(strategy_factor_names_field, 'default'):
                    strategy_factor_names_value = strategy_factor_names_field.default
                else:
                    strategy_factor_names_value = []
                
                # Use strategy name as the identifier
                strategy_name = strategy_name_value
                
                # Get or generate version from version_manager
                strategy_version = await version_manager.get_version("strategy", strategy_name)
                
                # Get full module source code
                strategy_code = dynamic_manager.get_full_module_source(strategy_cls)
                
                # Get strategy config from global config
                strategy_config_key = inflection.underscore(strategy_cls.__name__)
                strategy_config_dict = config.get(strategy_config_key, {})
                
                # Create strategy config
                strategy_config = StrategyConfig(
                    name=strategy_name_value,
                    description=strategy_description_value,
                    factor_names=strategy_factor_names_value,
                    version=strategy_version,
                    cls=strategy_cls,
                    config=strategy_config_dict,
                    instance=None,
                    code=strategy_code,
                )
                
                # Store strategy config
                strategy_configs[strategy_name] = strategy_config
                
                # Store in version history (by version string)
                if strategy_name not in self._strategy_history_versions:
                    self._strategy_history_versions[strategy_name] = {}
                self._strategy_history_versions[strategy_name][strategy_version] = strategy_config
                
                # Register version to version manager
                await version_manager.register_version("strategy", strategy_name, strategy_version)
                
                logger.info(f"| 📝 Registered strategy: {strategy_name} ({strategy_cls.__name__})")
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register strategy class {strategy_cls.__name__}: {e}")
                raise
        
        logger.info(f"| 🔍 Discovering {len(strategy_classes)} strategies from strategy modules")
        
        # Register each strategy class concurrently with a concurrency limit
        tasks = [
            register_strategy_class(strategy_cls) for strategy_cls in strategy_classes
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(strategy_classes)} strategies from strategy modules")
        
        return strategy_configs
    
    async def _load_from_code(self):
        """Load strategies from code (JSON file).
        
        JSON file content example:
        {
            "metadata": {
                "saved_at": str,  # "YYYY-MM-DD HH:MM:SS"
                "num_strategies": int,  # total strategy count
                "num_versions": int  # total version count
            },
            "strategies": {
                "strategy_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": {
                            "name": str,
                            "description": str,
                            "factor_names": List[str],
                            "version": str,
                            "cls": Type[Strategy],
                            "config": dict,
                            "instance": Strategy, # will be built when needed
                            "code": str
                        },
                        ...
                    }
                }
            }
        }
        
        Returns:
            Dict[str, StrategyConfig]: Dictionary mapping strategy names to their configs
        """
        
        strategy_configs: Dict[str, StrategyConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Strategy config file not found at {self.save_path}, skipping code-based loading")
            return strategy_configs
        
        # Load all strategy configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse strategy config JSON from {self.save_path}: {e}")
            return strategy_configs
        
        metadata = load_data.get("metadata", {})
        strategies_data = load_data.get("strategies", {})

        async def register_strategy_class(strategy_name: str, strategy_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, StrategyConfig], Optional[StrategyConfig]]]:
            """Load all versions for a single strategy from JSON."""
            try:
                current_version = strategy_data.get("current_version", "1.0.0")
                versions = strategy_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Strategy {strategy_name} has no versions")
                    return None
                
                version_map: Dict[str, StrategyConfig] = {}
                current_strategy_config: Optional[StrategyConfig] = None
                
                for _, version_data in versions.items():
                    strategy_config = StrategyConfig.model_validate(version_data)
                    version = strategy_config.version
                    version_map[version] = strategy_config
                    
                    if version == current_version:
                        current_strategy_config = strategy_config
                
                return strategy_name, version_map, current_strategy_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load strategy {strategy_name} from JSON: {e}")
                return None

        # Launch loading of each strategy concurrently with a concurrency limit
        tasks = [
            register_strategy_class(strategy_name, strategy_data) for strategy_name, strategy_data in strategies_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            strategy_name, version_map, current_strategy_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._strategy_history_versions[strategy_name] = version_map
            # Active config: the one corresponding to current_version
            if current_strategy_config is not None:
                strategy_configs[strategy_name] = current_strategy_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Strategy {strategy_name} current_version not found, using last available version")
                strategy_configs[strategy_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for strategy_config in version_map.values():
                await version_manager.register_version("strategy", strategy_name, strategy_config.version)
            
        logger.info(f"| 📂 Loaded {len(strategy_configs)} strategies from {self.save_path}")
        return strategy_configs
    
    async def build(self, strategy_config: StrategyConfig) -> StrategyConfig:
        """Create a strategy instance and store it.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            StrategyConfig: Strategy configuration with instance
        """
        if strategy_config.name in self._strategy_configs:
            existing_config = self._strategy_configs[strategy_config.name]
            if existing_config.instance is not None:
                return existing_config
        
        # Create new strategy instance
        try:
            # cls should already be loaded (either from module or from code/JSON)
            if strategy_config.cls is None:
                raise ValueError(f"Cannot create strategy {strategy_config.name}: no class provided. Class should be loaded during initialization.")
            
            # Instantiate strategy instance
            strategy_instance = strategy_config.cls(**strategy_config.config) if strategy_config.config else strategy_config.cls()
            
            # Initialize strategy if it has an initialize method
            if hasattr(strategy_instance, "initialize"):
                await strategy_instance.initialize()
            
            strategy_config.instance = strategy_instance
            
            # Store strategy metadata
            self._strategy_configs[strategy_config.name] = strategy_config
            
            logger.info(f"| 🔧 Strategy {strategy_config.name} created and stored")
            
            return strategy_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create strategy {strategy_config.name}: {e}")
            raise
    
    async def register(self, 
                       strategy_cls: Type[Strategy],
                       strategy_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None,
                       code: Optional[str] = None) -> StrategyConfig:
        """Register a strategy class or instance.
        
        This will:
        - Create (or reuse) a strategy instance
        - Create a `StrategyConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager`
        """
        
        try:
            if strategy_config_dict is None:
                # Fallback to global config by class name
                strategy_config_key = inflection.underscore(strategy_cls.__name__)
                strategy_config_dict = config.get(strategy_config_key, {})
            
            # Instantiate strategy immediately (register is a runtime operation)
            try:
                strategy_instance = strategy_cls(**strategy_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create strategy instance for {strategy_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate strategy {strategy_cls.__name__} with provided config: {e}")
            
            # Get strategy properties
            strategy_name = strategy_instance.name
            strategy_description = strategy_instance.description
            strategy_factor_names = strategy_instance.factor_names
            
            # Get or generate version from version_manager
            if version is None:
                strategy_version = await version_manager.get_version("strategy", strategy_name)
            else:
                strategy_version = version
                
            # Get strategy code
            if code is not None:
                # Use provided code string (e.g., when registering from code string)
                strategy_code = code
            else:
                # Try to extract code from the class
                strategy_code = dynamic_manager.get_source_code(strategy_cls)
                if not strategy_code:
                    logger.warning(f"| ⚠️ Strategy {strategy_name} is dynamic but source code cannot be extracted")
                    strategy_code = ""  # Ensure it's a string, not None
            
            # --- Build StrategyConfig ---
            strategy_config = StrategyConfig(
                name=strategy_name,
                description=strategy_description,
                factor_names=strategy_factor_names,
                version=strategy_version,
                cls=strategy_cls,
                config=strategy_config_dict or {},
                instance=strategy_instance,
                code=strategy_code or "",  # Ensure code is always a string
            )
            
            # --- Persist current config and history ---
            self._strategy_configs[strategy_name] = strategy_config
            
            # Store in dict-based history (for quick lookup by version)
            if strategy_name not in self._strategy_history_versions:
                self._strategy_history_versions[strategy_name] = {}
            self._strategy_history_versions[strategy_name][strategy_config.version] = strategy_config
            
            # Register version in version manager
            await version_manager.register_version("strategy", strategy_name, strategy_config.version)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📝 Registered strategy config: {strategy_name}: {strategy_config.version}")
            return strategy_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to register strategy: {e}")
            raise
    
    
    async def get(self, strategy_name: str) -> Strategy:
        """Get strategy configuration by name
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Strategy: Strategy instance or None if not found
        """
        strategy_config = self._strategy_configs.get(strategy_name)
        if strategy_config is None:
            return None
        return strategy_config.instance if strategy_config.instance is not None else None
    
    async def get_info(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get strategy info by name
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            StrategyConfig: Strategy info or None if not found
        """
        return self._strategy_configs.get(strategy_name)
    
    async def list(self) -> List[str]:
        """Get list of registered strategies
        
        Returns:
            List[str]: List of strategy names
        """
        return [name for name in self._strategy_configs.keys()]
    
    async def update(self, 
                     strategy_cls: Type[Strategy],
                     strategy_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None,
                     code: Optional[str] = None) -> StrategyConfig:
        """Update an existing strategy with new configuration and create a new version
        
        Args:
            strategy_cls: New strategy class with updated implementation
            strategy_config_dict: Configuration dict for strategy initialization
                   If None, will try to get from global config
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            StrategyConfig: Updated strategy configuration
        """
        try:
            if strategy_config_dict is None:
                # Fallback to global config by class name
                strategy_config_key = inflection.underscore(strategy_cls.__name__)
                strategy_config_dict = config.get(strategy_config_key, {})
            
            # Instantiate strategy immediately (update is a runtime operation)
            try:
                strategy_instance = strategy_cls(**strategy_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create strategy instance for {strategy_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate strategy {strategy_cls.__name__} with provided config: {e}")
            
            strategy_name = strategy_instance.name
            
            # Check if strategy exists
            original_config = self._strategy_configs.get(strategy_name)
            if original_config is None:
                raise ValueError(f"Strategy {strategy_name} not found. Use register() to register a new strategy.")
            
            strategy_description = strategy_instance.description
            strategy_factor_names = strategy_instance.factor_names
            
            # Determine new version from version_manager
            if new_version is None:
                # Get current version from version_manager and generate next patch version
                new_version = await version_manager.generate_next_version("strategy", strategy_name, "patch")
            
            # Get strategy code
            if code is not None:
                # Use provided code string (e.g., when updating from code string)
                strategy_code = code
            else:
                # Try to extract code from the class
                strategy_code = dynamic_manager.get_source_code(strategy_cls)
                if not strategy_code:
                    logger.warning(f"| ⚠️ Strategy {strategy_name} is dynamic but source code cannot be extracted")
                    strategy_code = ""  # Ensure it's a string, not None
            
            # --- Build StrategyConfig ---
            updated_config = StrategyConfig(
                name=strategy_name,  # Keep same name
                description=strategy_description,
                factor_names=strategy_factor_names,
                version=new_version,
                cls=strategy_cls,
                config=strategy_config_dict or {},
                instance=strategy_instance,
                code=strategy_code or "",  # Ensure code is always a string
            )
            
            # Update the strategy config (replaces current version)
            self._strategy_configs[strategy_name] = updated_config
            
            # Store in version history
            if strategy_name not in self._strategy_history_versions:
                self._strategy_history_versions[strategy_name] = {}
            self._strategy_history_versions[strategy_name][updated_config.version] = updated_config
            
            # Register new version record to version manager
            await version_manager.register_version(
                "strategy", 
                strategy_name, 
                new_version,
                description=description or f"Updated from {original_config.version}"
            )
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 🔄 Updated strategy {strategy_name} from v{original_config.version} to v{new_version}")
            return updated_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to update strategy: {e}")
            raise
    
    async def copy(self, 
                  strategy_name: str,
                  new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, 
                  new_config: Optional[Dict[str, Any]] = None) -> StrategyConfig:
        """Copy an existing strategy configuration
        
        Args:
            strategy_name: Name of the strategy to copy
            new_name: New name for the copied strategy. If None, uses original name.
            new_version: New version for the copied strategy. If None, increments version.
            new_config: New configuration dict for the copied strategy. If None, uses original config.
            
        Returns:
            StrategyConfig: New strategy configuration
        """
        try:
            original_config = self._strategy_configs.get(strategy_name)
            if original_config is None:
                raise ValueError(f"Strategy {strategy_name} not found")
            
            if original_config.cls is None:
                raise ValueError(f"Cannot copy strategy {strategy_name}: no class provided")
            
            # Determine new name
            if new_name is None:
                new_name = strategy_name
            
            # Prepare config dict (merge original config with new config)
            strategy_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # Merge new config into original config
                strategy_config_dict.update(new_config)
            
            # Instantiate strategy instance (copy is a runtime operation)
            try:
                strategy_instance = original_config.cls(**strategy_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create strategy instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate strategy {original_config.cls.__name__} with provided config: {e}")
            
            # Apply name override if provided (after instantiation)
            if new_name != strategy_name:
                strategy_instance.name = new_name
            
            strategy_description = strategy_instance.description
            strategy_factor_names = strategy_instance.factor_names
            
            # Determine new version from version_manager
            if new_version is None:
                if new_name == strategy_name:
                    # If copying with same name, get next version from version_manager
                    new_version = await version_manager.generate_next_version("strategy", new_name, "patch")
                else:
                    # If copying with different name, get or generate version for new name
                    new_version = await version_manager.get_version("strategy", new_name)
            
            # Get strategy code - use original config's code if available
            strategy_code = original_config.code if original_config.code else ""
            if not strategy_code:
                # Fallback: try to extract code from the class
                strategy_code = dynamic_manager.get_source_code(original_config.cls)
                if not strategy_code:
                    logger.warning(f"| ⚠️ Strategy {new_name} is dynamic but source code cannot be extracted")
                    strategy_code = ""  # Ensure it's a string, not None
            
            # --- Build StrategyConfig ---
            new_config = StrategyConfig(
                name=new_name,
                description=strategy_description,
                factor_names=strategy_factor_names,
                version=new_version,
                cls=original_config.cls,
                config=strategy_config_dict,
                instance=strategy_instance,
                code=strategy_code or "",  # Ensure code is always a string
            )
            
            # Register new strategy
            self._strategy_configs[new_name] = new_config
            
            # Store in version history
            if new_name not in self._strategy_history_versions:
                self._strategy_history_versions[new_name] = {}
            self._strategy_history_versions[new_name][new_version] = new_config
            
            # Register version record to version manager
            await version_manager.register_version(
                "strategy", 
                new_name, 
                new_version,
                description=f"Copied from {strategy_name}@{original_config.version}"
            )
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📋 Copied strategy {strategy_name}@{original_config.version} to {new_name}@{new_version}")
            return new_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to copy strategy: {e}")
            raise
    
    async def unregister(self, strategy_name: str) -> bool:
        """Unregister a strategy
        
        Args:
            strategy_name: Name of the strategy to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if strategy_name not in self._strategy_configs:
            logger.warning(f"| ⚠️ Strategy {strategy_name} not found")
            return False
        
        strategy_config = self._strategy_configs[strategy_name]
        
        # Remove from configs
        del self._strategy_configs[strategy_name]

        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered strategy {strategy_name}@{strategy_config.version}")
        return True
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all strategy configurations with version history to JSON.
        
        Only saves basic configuration fields (name, description, factor_names, version, config, etc.).
        Instance is not saved as it's runtime state and will be recreated via build() on load.
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            # Ensure parent directory exists
            parent_dir = os.path.dirname(file_path)
            if parent_dir:  # Only create if there's a directory component
                os.makedirs(parent_dir, exist_ok=True)
            
            # Prepare save data - save all versions for each strategy
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_strategies": len(self._strategy_configs),
                    "num_versions": sum(len(versions) for versions in self._strategy_history_versions.values()),
                },
                "strategies": {}
            }
            
            for strategy_name, version_map in self._strategy_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, strategy_config in version_map.items():
                        # Exclude 'cls' and 'instance' fields during serialization
                        config_dict = strategy_config.model_dump(exclude={"cls", "instance"})
                        versions_data[strategy_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if strategy_name in self._strategy_configs:
                        current_config = self._strategy_configs[strategy_name]
                        if current_config is not None:
                            current_version = current_config.version
                    
                    # If not found in active configs, use latest version from history
                    if current_version is None and version_map:
                        # Find latest version by comparing version strings
                        latest_version_str = None
                        for version_str in version_map.keys():
                            if latest_version_str is None:
                                latest_version_str = version_str
                            elif version_manager.compare_versions(version_str, latest_version_str) > 0:
                                latest_version_str = version_str
                        current_version = latest_version_str
                    
                    save_data["strategies"][strategy_name] = {
                        "versions": versions_data,
                        "current_version": current_version
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize strategy {strategy_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._strategy_configs)} strategies with version history to {file_path}")
            return str(file_path)
    
    async def restore(self, strategy_name: str, version: str, auto_initialize: bool = True) -> Optional[StrategyConfig]:
        """Restore a specific version of a strategy from history
        
        Args:
            strategy_name: Name of the strategy
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored strategy
            
        Returns:
            StrategyConfig of the restored version, or None if not found
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if strategy_name in self._strategy_history_versions:
            version_config = self._strategy_history_versions[strategy_name].get(version)
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for strategy {strategy_name}")
            return None
        
        # Create a copy to avoid modifying the history
        restored_config = StrategyConfig(**version_config.model_dump())
        
        # Set as current active config
        self._strategy_configs[strategy_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("strategy", strategy_name)
        if version_history:
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("strategy", strategy_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("strategy", strategy_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored strategy {strategy_name} to version {version}")
        return restored_config
    
    async def save_contract(self, strategy_names: Optional[List[str]] = None):
        """Save the contract for strategies"""
        contract = []
        # Use provided strategy names or all successfully initialized strategies
        names_to_save = strategy_names if strategy_names is not None else list(self._strategy_configs.keys())
        
        for index, strategy_name in enumerate(names_to_save):
            strategy_info = await self.get_info(strategy_name)
            if strategy_info:
                contract.append(f"{index + 1:04d}\nName: {strategy_info.name}\nDescription: {strategy_info.description}\nFactor Names: {', '.join(strategy_info.factor_names)}\n")
        
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} strategies contract to {self.contract_path}")
    
    async def load_contract(self) -> str:
        """Load the contract for strategies"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
    async def cleanup(self):
        """Cleanup all active strategies."""
        try:
            # Clear all strategy configs and version history
            self._strategy_configs.clear()
            self._strategy_history_versions.clear()
                
            logger.info("| 🧹 Strategy context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during strategy context manager cleanup: {e}")

