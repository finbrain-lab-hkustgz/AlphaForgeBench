"""Factor Context Manager for managing factor lifecycle and resources with lazy loading."""
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
from src.factor.types import Factor, FactorConfig
from src.version import version_manager
from src.dynamic import dynamic_manager

class FactorContextManager(BaseModel):
    """Global context manager for all factors with lazy loading support."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the factors")
    save_path: str = Field(default=None, description="The path to save the factors")
    contract_path: str = Field(default=None, description="The path to save the factor contract")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 contract_path: Optional[str] = None,
                 **kwargs):
        """Initialize the factor context manager.
        
        Args:
            base_dir: Base directory for storing factor data
            save_path: Path to save factor configurations
        """
        super().__init__(**kwargs)
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "factor"))
        logger.info(f"| 📁 Factor context manager base directory: {self.base_dir}.")    
        os.makedirs(self.base_dir, exist_ok=True)
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "factor.json")
        logger.info(f"| 📁 Factor context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Factor context manager contract path: {self.contract_path}.")

        self._factor_configs: Dict[str, FactorConfig] = {}  # Current active configs (latest version)
        # Factor version history, e.g., {"factor_name": {"1.0.0": FactorConfig, "1.0.1": FactorConfig}}
        self._factor_history_versions: Dict[str, Dict[str, FactorConfig]] = {}
        
        self._cleanup_registered = False
        
    async def initialize(self, factor_names: Optional[List[str]] = None):
        """Initialize the factor context manager."""
        
        # Register factor-related symbols for auto-injection in dynamic code
        dynamic_manager.register_symbol("Factor", Factor)
        dynamic_manager.register_symbol("FactorConfig", FactorConfig)
        
        # Register factor context provider for automatic import injection
        def factor_context_provider():
            """Provide factor-related imports for dynamic factor classes."""
            return {
                "Factor": Factor,
                "FactorConfig": FactorConfig,
            }
        dynamic_manager.register_context_provider("factor", factor_context_provider)
        
        # Load factors from module
        module_factor_configs: Dict[str, FactorConfig] = await self._load_from_module()
        
        # Load factors from code (JSON file)
        code_factor_configs: Dict[str, FactorConfig] = await self._load_from_code()
        
        # Merge code configs with module configs, only override if code version is strictly greater
        factor_configs = {}
        factor_configs.update(module_factor_configs)
        
        for factor_name, code_config in code_factor_configs.items():
            if factor_name in factor_configs:
                module_config = factor_configs[factor_name]
                # Compare versions: only override if code version is strictly greater
                if version_manager.compare_versions(code_config.version, module_config.version) > 0:
                    logger.info(f"| 🔄 Overriding factor {factor_name} from module (v{module_config.version}) with code version (v{code_config.version})")
                    factor_configs[factor_name] = code_config
                else:
                    logger.info(f"| 📌 Keeping factor {factor_name} from module (v{module_config.version}), code version (v{code_config.version}) is not greater")
                    # If versions are equal, update the history with module config (which has real class, not dynamic)
                    if version_manager.compare_versions(code_config.version, module_config.version) == 0:
                        # Replace the code config in history with module config to preserve real class reference
                        if factor_name in self._factor_history_versions:
                            self._factor_history_versions[factor_name][module_config.version] = module_config
            else:
                # New factor from code, add it
                factor_configs[factor_name] = code_config
        
        # Filter factors by names if provided
        if factor_names is not None:
            factor_configs = {name: factor_configs[name] for name in factor_names if name in factor_configs}
        
        # Build all factors concurrently with a concurrency limit
        factor_names_list = list(factor_configs.keys())
        tasks = [
            self.build(factor_configs[name]) for name in factor_names_list
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for factor_name, result in zip(factor_names_list, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize factor {factor_name}: {result}")
                continue
            self._factor_configs[factor_name] = result
            logger.info(f"| 🔧 Factor {factor_name} initialized")
        
        # Save factor configs to json file
        await self.save_to_json()
        # Save contract to file
        await self.save_contract(factor_names=factor_names_list)
        
        # Register cleanup callback
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True
        
        logger.info(f"| ✅ Factors initialization completed")
        
    async def _load_from_module(self):
        """Load factors from module (factors module).
        
        Automatically discovers all Factor classes from src/factor/factors module.
        
        Returns:
            Dict[str, FactorConfig]: Dictionary mapping factor names to their configs
        """
        factor_configs: Dict[str, FactorConfig] = {}
        
        # Discover all Factor classes from factors module
        factor_classes = []
        
        # Import the factors package
        try:
            factors_package = importlib.import_module("src.factor.factors")
        except ImportError as e:
            logger.error(f"| ❌ Failed to import factors package: {e}")
            return factor_configs
        
        # Iterate through all modules in the factors package
        for importer, modname, ispkg in pkgutil.iter_modules(factors_package.__path__, factors_package.__name__ + "."):
            if ispkg:
                continue  # Skip sub-packages
            
            try:
                # Import the module
                module = importlib.import_module(modname)
                
                # Find all Factor subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, Factor) and 
                        obj is not Factor and 
                        obj.__module__ == modname):
                        factor_classes.append(obj)
                        logger.debug(f"| 🔍 Discovered factor class: {name} from {modname}")
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to import module {modname}: {e}")
                continue
        
        async def register_factor_class(factor_cls: Type[Factor]):
            """Register a factor class.
            
            Args:
                factor_cls: Factor class to register
            """
            try:
                # Get factor properties from factor class
                factor_type = factor_cls.model_fields.get('type', None)
                if factor_type and hasattr(factor_type, 'default'):
                    factor_type_value = factor_type.default
                else:
                    factor_type_value = inflection.underscore(factor_cls.__name__)
                
                factor_expression = factor_cls.model_fields.get('expression', None)
                if factor_expression and hasattr(factor_expression, 'default'):
                    factor_expression_value = factor_expression.default
                else:
                    factor_expression_value = ""
                
                factor_description = factor_cls.model_fields.get('description', None)
                if factor_description and hasattr(factor_description, 'default'):
                    factor_description_value = factor_description.default
                else:
                    factor_description_value = f"{factor_cls.__name__} factor"
                
                factor_names = factor_cls.model_fields.get('names', None)
                if factor_names and hasattr(factor_names, 'default'):
                    factor_names_value = factor_names.default
                else:
                    factor_names_value = [inflection.underscore(factor_cls.__name__)]
                
                # Use factor type as the name
                factor_name = factor_type_value
                
                # Get or generate version from version_manager
                factor_version = await version_manager.get_version("factor", factor_name)
                
                # Get full module source code
                factor_code = dynamic_manager.get_full_module_source(factor_cls)
                
                # Get factor config from global config
                factor_config_key = inflection.underscore(factor_cls.__name__)
                factor_config_dict = config.get(factor_config_key, {})
                
                # Create factor config
                factor_config = FactorConfig(
                    type=factor_type_value,
                    expression=factor_expression_value,
                    description=factor_description_value,
                    names=factor_names_value,
                    version=factor_version,
                    cls=factor_cls,
                    config=factor_config_dict,
                    instance=None,
                    code=factor_code,
                )
                
                # Store factor config
                factor_configs[factor_name] = factor_config
                
                # Store in version history (by version string)
                if factor_name not in self._factor_history_versions:
                    self._factor_history_versions[factor_name] = {}
                self._factor_history_versions[factor_name][factor_version] = factor_config
                
                # Register version to version manager
                await version_manager.register_version("factor", factor_name, factor_version)
                
                logger.info(f"| 📝 Registered factor: {factor_name} ({factor_cls.__name__})")
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register factor class {factor_cls.__name__}: {e}")
                raise
        
        logger.info(f"| 🔍 Discovering {len(factor_classes)} factors from factors module")
        
        # Register each factor class concurrently with a concurrency limit
        tasks = [
            register_factor_class(factor_cls) for factor_cls in factor_classes
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(factor_classes)} factors from factors module")
        
        return factor_configs
    
    async def _load_from_code(self):
        """Load factors from code (JSON file).
        
        JSON file content example:
        {
            "metadata": {
                "saved_at": str,  # "YYYY-MM-DD HH:MM:SS"
                "num_factors": int,  # total factor count
                "num_versions": int  # total version count
            },
            "factors": {
                "factor_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": {
                            "type": str,
                            "expression": str,
                            "description": str,
                            "names": List[str],
                            "version": str,
                            "cls": Type[Factor],
                            "config": dict,
                            "instance": Factor, # will be built when needed
                            "code": str
                        },
                        ...
                    }
                }
            }
        }
        
        Returns:
            Dict[str, FactorConfig]: Dictionary mapping factor names to their configs
        """
        
        factor_configs: Dict[str, FactorConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Factor config file not found at {self.save_path}, skipping code-based loading")
            return factor_configs
        
        # Load all factor configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse factor config JSON from {self.save_path}: {e}")
            return factor_configs
        
        metadata = load_data.get("metadata", {})
        factors_data = load_data.get("factors", {})

        async def register_factor_class(factor_name: str, factor_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, FactorConfig], Optional[FactorConfig]]]:
            """Load all versions for a single factor from JSON."""
            try:
                current_version = factor_data.get("current_version", "1.0.0")
                versions = factor_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Factor {factor_name} has no versions")
                    return None
                
                version_map: Dict[str, FactorConfig] = {}
                current_factor_config: Optional[FactorConfig] = None
                
                for _, version_data in versions.items():
                    factor_config = FactorConfig.model_validate(version_data)
                    version = factor_config.version
                    version_map[version] = factor_config
                    
                    if version == current_version:
                        current_factor_config = factor_config
                
                return factor_name, version_map, current_factor_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load factor {factor_name} from JSON: {e}")
                return None

        # Launch loading of each factor concurrently with a concurrency limit
        tasks = [
            register_factor_class(factor_name, factor_data) for factor_name, factor_data in factors_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            factor_name, version_map, current_factor_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._factor_history_versions[factor_name] = version_map
            # Active config: the one corresponding to current_version
            if current_factor_config is not None:
                factor_configs[factor_name] = current_factor_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Factor {factor_name} current_version not found, using last available version")
                factor_configs[factor_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for factor_config in version_map.values():
                await version_manager.register_version("factor", factor_name, factor_config.version)
            
        logger.info(f"| 📂 Loaded {len(factor_configs)} factors from {self.save_path}")
        return factor_configs
    
    async def build(self, factor_config: FactorConfig) -> FactorConfig:
        """Create a factor instance and store it.
        
        Args:
            factor_config: Factor configuration
            
        Returns:
            FactorConfig: Factor configuration with instance
        """
        if factor_config.type in self._factor_configs:
            existing_config = self._factor_configs[factor_config.type]
            if existing_config.instance is not None:
                return existing_config
        
        # Create new factor instance
        try:
            # cls should already be loaded (either from registry or from code/JSON)
            if factor_config.cls is None:
                raise ValueError(f"Cannot create factor {factor_config.type}: no class provided. Class should be loaded during initialization.")
            
            # Instantiate factor instance
            factor_instance = factor_config.cls(**factor_config.config) if factor_config.config else factor_config.cls()
            
            # Initialize factor if it has an initialize method
            if hasattr(factor_instance, "initialize"):
                await factor_instance.initialize()
            
            factor_config.instance = factor_instance
            
            # Store factor metadata
            self._factor_configs[factor_config.type] = factor_config
            
            logger.info(f"| 🔧 Factor {factor_config.type} created and stored")
            
            return factor_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create factor {factor_config.type}: {e}")
            raise
    
    async def register(self, 
                       factor_cls: Type[Factor],
                       factor_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None,
                       code: Optional[str] = None) -> FactorConfig:
        """Register a factor class or instance.
        
        This will:
        - Create (or reuse) a factor instance
        - Create a `FactorConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager`
        """
        
        try:
            if factor_config_dict is None:
                # Fallback to global config by class name
                factor_config_key = inflection.underscore(factor_cls.__name__)
                factor_config_dict = config.get(factor_config_key, {})
            
            # Instantiate factor immediately (register is a runtime operation)
            try:
                factor_instance = factor_cls(**factor_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create factor instance for {factor_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate factor {factor_cls.__name__} with provided config: {e}")
            
            # Get factor properties
            factor_type = factor_instance.type
            factor_expression = factor_instance.expression
            factor_description = factor_instance.description
            factor_names = factor_instance.names
            
            # Get or generate version from version_manager
            if version is None:
                factor_version = await version_manager.get_version("factor", factor_type)
            else:
                factor_version = version
                
            # Get factor code
            if code is not None:
                # Use provided code string (e.g., when registering from code string)
                factor_code = code
            else:
                # Try to extract code from the class
                factor_code = dynamic_manager.get_source_code(factor_cls)
                if not factor_code:
                    logger.warning(f"| ⚠️ Factor {factor_type} is dynamic but source code cannot be extracted")
                    factor_code = ""  # Ensure it's a string, not None
            
            # --- Build FactorConfig ---
            factor_config = FactorConfig(
                type=factor_type,
                expression=factor_expression,
                description=factor_description,
                names=factor_names,
                version=factor_version,
                cls=factor_cls,
                config=factor_config_dict or {},
                instance=factor_instance,
                code=factor_code or "",  # Ensure code is always a string
            )
            
            # --- Persist current config and history ---
            self._factor_configs[factor_type] = factor_config
            
            # Store in dict-based history (for quick lookup by version)
            if factor_type not in self._factor_history_versions:
                self._factor_history_versions[factor_type] = {}
            self._factor_history_versions[factor_type][factor_config.version] = factor_config
            
            # Register version in version manager
            await version_manager.register_version("factor", factor_type, factor_config.version)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📝 Registered factor config: {factor_type}: {factor_config.version}")
            return factor_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to register factor: {e}")
            raise
    
    
    async def get(self, factor_name: str) -> Factor:
        """Get factor configuration by name
        
        Args:
            factor_name: Factor name (type)
            
        Returns:
            Factor: Factor instance or None if not found
        """
        factor_config = self._factor_configs.get(factor_name)
        if factor_config is None:
            return None
        return factor_config.instance if factor_config.instance is not None else None
    
    async def get_info(self, factor_name: str) -> Optional[FactorConfig]:
        """Get factor info by name
        
        Args:
            factor_name: Factor name (type)
            
        Returns:
            FactorConfig: Factor info or None if not found
        """
        return self._factor_configs.get(factor_name)
    
    async def list(self) -> List[str]:
        """Get list of registered factors
        
        Returns:
            List[str]: List of factor names (types)
        """
        return [name for name in self._factor_configs.keys()]
    
    async def update(self, 
                     factor_cls: Type[Factor],
                     factor_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> FactorConfig:
        """Update an existing factor with new configuration and create a new version
        
        Args:
            factor_cls: New factor class with updated implementation
            factor_config_dict: Configuration dict for factor initialization
                   If None, will try to get from global config
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            FactorConfig: Updated factor configuration
        """
        try:
            if factor_config_dict is None:
                # Fallback to global config by class name
                factor_config_key = inflection.underscore(factor_cls.__name__)
                factor_config_dict = config.get(factor_config_key, {})
            
            # Instantiate factor immediately (update is a runtime operation)
            try:
                factor_instance = factor_cls(**factor_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create factor instance for {factor_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate factor {factor_cls.__name__} with provided config: {e}")
            
            factor_type = factor_instance.type
            
            # Check if factor exists
            original_config = self._factor_configs.get(factor_type)
            if original_config is None:
                raise ValueError(f"Factor {factor_type} not found. Use register() to register a new factor.")
            
            factor_expression = factor_instance.expression
            factor_description = factor_instance.description
            factor_names = factor_instance.names
            
            # Determine new version from version_manager
            if new_version is None:
                # Get current version from version_manager and generate next patch version
                new_version = await version_manager.generate_next_version("factor", factor_type, "patch")
            
            # Get factor code
            factor_code = dynamic_manager.get_source_code(factor_cls)
            if not factor_code:
                logger.warning(f"| ⚠️ Factor {factor_type} is dynamic but source code cannot be extracted")
            
            # --- Build FactorConfig ---
            updated_config = FactorConfig(
                type=factor_type,  # Keep same type
                expression=factor_expression,
                description=factor_description,
                names=factor_names,
                version=new_version,
                cls=factor_cls,
                config=factor_config_dict or {},
                instance=factor_instance,
                code=factor_code,
            )
            
            # Update the factor config (replaces current version)
            self._factor_configs[factor_type] = updated_config
            
            # Store in version history
            if factor_type not in self._factor_history_versions:
                self._factor_history_versions[factor_type] = {}
            self._factor_history_versions[factor_type][updated_config.version] = updated_config
            
            # Register new version record to version manager
            await version_manager.register_version(
                "factor", 
                factor_type, 
                new_version,
                description=description or f"Updated from {original_config.version}"
            )
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 🔄 Updated factor {factor_type} from v{original_config.version} to v{new_version}")
            return updated_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to update factor: {e}")
            raise
    
    async def copy(self, 
                  factor_name: str,
                  new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, 
                  new_config: Optional[Dict[str, Any]] = None) -> FactorConfig:
        """Copy an existing factor configuration
        
        Args:
            factor_name: Name (type) of the factor to copy
            new_name: New name (type) for the copied factor. If None, uses original name.
            new_version: New version for the copied factor. If None, increments version.
            new_config: New configuration dict for the copied factor. If None, uses original config.
            
        Returns:
            FactorConfig: New factor configuration
        """
        try:
            original_config = self._factor_configs.get(factor_name)
            if original_config is None:
                raise ValueError(f"Factor {factor_name} not found")
            
            if original_config.cls is None:
                raise ValueError(f"Cannot copy factor {factor_name}: no class provided")
            
            # Determine new name
            if new_name is None:
                new_name = factor_name
            
            # Prepare config dict (merge original config with new config)
            factor_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # Merge new config into original config
                factor_config_dict.update(new_config)
            
            # Instantiate factor instance (copy is a runtime operation)
            try:
                factor_instance = original_config.cls(**factor_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create factor instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate factor {original_config.cls.__name__} with provided config: {e}")
            
            # Apply name override if provided (after instantiation)
            if new_name != factor_name:
                factor_instance.type = new_name
            
            factor_expression = factor_instance.expression
            factor_description = factor_instance.description
            factor_names = factor_instance.names
            
            # Determine new version from version_manager
            if new_version is None:
                if new_name == factor_name:
                    # If copying with same name, get next version from version_manager
                    new_version = await version_manager.generate_next_version("factor", new_name, "patch")
                else:
                    # If copying with different name, get or generate version for new name
                    new_version = await version_manager.get_version("factor", new_name)
            
            # Get factor code
            factor_code = dynamic_manager.get_source_code(original_config.cls)
            if not factor_code:
                logger.warning(f"| ⚠️ Factor {new_name} is dynamic but source code cannot be extracted")
            
            # --- Build FactorConfig ---
            new_config = FactorConfig(
                type=new_name,
                expression=factor_expression,
                description=factor_description,
                names=factor_names,
                version=new_version,
                cls=original_config.cls,
                config=factor_config_dict,
                instance=factor_instance,
                code=factor_code,
            )
            
            # Register new factor
            self._factor_configs[new_name] = new_config
            
            # Store in version history
            if new_name not in self._factor_history_versions:
                self._factor_history_versions[new_name] = {}
            self._factor_history_versions[new_name][new_version] = new_config
            
            # Register version record to version manager
            await version_manager.register_version(
                "factor", 
                new_name, 
                new_version,
                description=f"Copied from {factor_name}@{original_config.version}"
            )
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📋 Copied factor {factor_name}@{original_config.version} to {new_name}@{new_version}")
            return new_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to copy factor: {e}")
            raise
    
    async def unregister(self, factor_name: str) -> bool:
        """Unregister a factor
        
        Args:
            factor_name: Name (type) of the factor to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if factor_name not in self._factor_configs:
            logger.warning(f"| ⚠️ Factor {factor_name} not found")
            return False
        
        factor_config = self._factor_configs[factor_name]
        
        # Remove from configs
        del self._factor_configs[factor_name]

        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered factor {factor_name}@{factor_config.version}")
        return True
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all factor configurations with version history to JSON.
        
        Only saves basic configuration fields (type, expression, description, names, version, config, etc.).
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
            
            # Prepare save data - save all versions for each factor
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_factors": len(self._factor_configs),
                    "num_versions": sum(len(versions) for versions in self._factor_history_versions.values()),
                },
                "factors": {}
            }
            
            for factor_name, version_map in self._factor_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, factor_config in version_map.items():
                        config_dict = factor_config.model_dump()
                        versions_data[factor_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if factor_name in self._factor_configs:
                        current_config = self._factor_configs[factor_name]
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
                    
                    save_data["factors"][factor_name] = {
                        "versions": versions_data,
                        "current_version": current_version
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize factor {factor_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._factor_configs)} factors with version history to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load factor configurations with version history from JSON.
        
        Loads basic configuration only (instance is not saved, must be created via build()).
        Only the latest version will be instantiated by default if auto_initialize=True.
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically create instance via build() after loading
            
        Returns:
            True if loaded successfully, False otherwise
        """
        
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Factor file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                factors_data = load_data.get("factors", {})
                loaded_count = 0
                
                for factor_name, factor_data in factors_data.items():
                    try:
                        # Expected format: multiple versions stored as a dict {version_str: config_dict}
                        versions_data = factor_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Factor {factor_name} has invalid format for 'versions' (expected dict), skipping")
                            continue
                        
                        current_version_str = factor_data.get("current_version")
                        
                        # Load all versions
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for version_str, config_dict in versions_data.items():
                            # Ensure version field is present
                            if "version" not in config_dict:
                                config_dict["version"] = version_str
                            
                            try:
                                factor_config = FactorConfig.model_validate(config_dict)
                                version_configs.append(factor_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load factor config for {factor_name}@{version_str}: {e}")
                                continue
                            
                            # Track latest version
                            if latest_config is None or (
                                current_version_str and factor_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    version_manager.compare_versions(factor_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = factor_config
                                latest_version = factor_config.version
                        
                        # Store all versions in history (dict-based)
                        self._factor_history_versions[factor_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }
                        
                        # Only set latest version as active
                        if latest_config:
                            self._factor_configs[factor_name] = latest_config
                            
                            # Register all versions to version manager (only version records)
                            for factor_config in version_configs:
                                await version_manager.register_version("factor", factor_name, factor_config.version)
                            
                            # Create instance if requested (instance is not saved in JSON, must be created via build)
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load factor {factor_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} factors with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load factors from {file_path}: {e}")
                return False
    
    async def restore(self, factor_name: str, version: str, auto_initialize: bool = True) -> Optional[FactorConfig]:
        """Restore a specific version of a factor from history
        
        Args:
            factor_name: Name (type) of the factor
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored factor
            
        Returns:
            FactorConfig of the restored version, or None if not found
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if factor_name in self._factor_history_versions:
            version_config = self._factor_history_versions[factor_name].get(version)
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for factor {factor_name}")
            return None
        
        # Create a copy to avoid modifying the history
        restored_config = FactorConfig(**version_config.model_dump())
        
        # Set as current active config
        self._factor_configs[factor_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("factor", factor_name)
        if version_history:
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("factor", factor_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("factor", factor_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored factor {factor_name} to version {version}")
        return restored_config
    
    async def save_contract(self, factor_names: Optional[List[str]] = None):
        """Save the contract for factors"""
        contract = []
        if factor_names is not None:
            for index, factor_name in enumerate(factor_names):
                factor_info = await self.get_info(factor_name)
                if factor_info:
                    contract.append(f"{index + 1:04d}\nType: {factor_info.type}\nExpression: {factor_info.expression}\nDescription: {factor_info.description}\nNames: {', '.join(factor_info.names)}\n")
        else:
            for index, factor_name in enumerate(self._factor_configs.keys()):
                factor_info = await self.get_info(factor_name)
                if factor_info:
                    contract.append(f"{index + 1:04d}\nType: {factor_info.type}\nExpression: {factor_info.expression}\nDescription: {factor_info.description}\nNames: {', '.join(factor_info.names)}\n")
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} factors contract to {self.contract_path}")
        
    async def load_contract(self) -> str:
        """Load the contract for factors"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
    async def cleanup(self):
        """Cleanup all active factors."""
        try:
            # Clear all factor configs and version history
            self._factor_configs.clear()
            self._factor_history_versions.clear()
                
            logger.info("| 🧹 Factor context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during factor context manager cleanup: {e}")

