"""Factor Server

Server implementation for the Factor Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
import os
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd

from src.logger import logger
from src.config import config
from src.factor.context import FactorContextManager
from src.factor.types import Factor, FactorConfig
from src.utils import assemble_project_path
from src.dynamic import dynamic_manager

class FactorManager(BaseModel):
    """Factor Manager for managing factor registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the factors")
    save_path: str = Field(default=None, description="The path to save the factors")
    contract_path: str = Field(default=None, description="The path to save the factor contract")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the Factor Manager."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, FactorConfig] = {}  # factor_name -> FactorConfig

        
    async def initialize(self, factor_names: Optional[List[str]] = None):
        """Initialize factors by names using factor context manager with concurrent support.
        
        Args:
            factor_names: List of factor names to initialize. If None, initialize all registered factors.
        """
        
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "factor"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "factor.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Factor Manager base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")
        
        # Initialize factor context manager
        self.factor_context_manager = FactorContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
        )
        await self.factor_context_manager.initialize(factor_names=factor_names)
        
        logger.info("| ✅ Factors initialization completed")
        
    async def get_contract(self) -> str:
        """Get the contract for all factors"""
        return await self.factor_context_manager.load_contract()
    
    async def register(self, 
                       factor: Union[str, Factor, Type[Factor]],
                       config: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> FactorConfig:
        """Register a factor from code string, class, or instance asynchronously.
        
        Args:
            factor: Factor code string, Factor class, or Factor instance to register
            config: Configuration dict for factor initialization (required when factor is a class or code)
            override: Whether to override existing registration
            version: Optional version string
            
        Returns:
            FactorConfig: Factor configuration
            
        Raises:
            ValueError: If factor code cannot be loaded or factor type is invalid
        """
        # Handle code string - load factor class dynamically
        if isinstance(factor, str):
            logger.info(f"| 📝 Loading factor from code...")
            factor_cls = await self._load_factor_from_code(factor)
            # Use empty config if not provided
            if config is None:
                config = {}
            # Pass the code string to context manager
            factor_config = await self.factor_context_manager.register(
                factor_cls, 
                factor_config_dict=config, 
                override=override,
                version=version,
                code=factor  # Pass the original code string
            )
        # Handle Factor instance
        elif isinstance(factor, Factor):
            factor_cls = type(factor)
            # Use instance's config if config not provided
            if config is None:
                config = {}
            factor_config = await self.factor_context_manager.register(
                factor_cls, 
                factor_config_dict=config, 
                override=override,
                version=version
            )
        # Handle Factor class
        else:
            factor_cls = factor
            factor_config = await self.factor_context_manager.register(
                factor_cls, 
                factor_config_dict=config, 
                override=override,
                version=version
            )
        self._registered_configs[factor_config.type] = factor_config
        logger.info(f"| ✅ Factor registered: {factor_config.type}")
        return factor_config
    
    async def _load_factor_from_code(self, code: str) -> Type[Factor]:
        """
        Load factor class from code string using dynamic_manager.
        
        This follows the same pattern as FactorConfig.model_validate.
        
        Args:
            code: Factor code string
            
        Returns:
            Factor class
            
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
                factor_cls = dynamic_manager.load_class(
                    code,
                    class_name=class_name,
                    base_class=Factor,
                    context="factor"
                )
            else:
                # Fallback: let load_class find the class automatically using base_class
                factor_cls = dynamic_manager.load_class(
                    code,
                    class_name=None,
                    base_class=Factor,
                    context="factor"
                )
        except Exception as e:
            raise ValueError(f"Failed to load factor class from code: {e}")
        
        if factor_cls is None:
            raise ValueError(f"Failed to load factor class from code")
        
        return factor_cls
    
    async def list(self) -> List[str]:
        """List all registered factors
        
        Returns:
            List[str]: List of factor names (types)
        """
        return await self.factor_context_manager.list()
    
    
    async def get(self, factor_name: str) -> Factor:
        """Get factor instance by name
        
        Args:
            factor_name: Factor name (type)
            
        Returns:
            Factor: Factor instance or None if not found
        """
        factor = await self.factor_context_manager.get(factor_name)
        return factor
    
    async def get_info(self, factor_name: str) -> Optional[FactorConfig]:
        """Get factor configuration by name
        
        Args:
            factor_name: Factor name (type)
            
        Returns:
            FactorConfig: Factor configuration or None if not found
        """
        return await self.factor_context_manager.get_info(factor_name)
    
    async def cleanup(self):
        """Cleanup all factors"""
        await self.factor_context_manager.cleanup()
        self._registered_configs.clear()
    
    async def update(self, 
                     factor: Union[str, Factor, Type[Factor]], 
                     config: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> FactorConfig:
        """Update an existing factor with new configuration and create a new version
        
        Args:
            factor: New factor code string, class, or instance with updated implementation
            config: Configuration dict for factor initialization (required when factor is a class or code)
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            FactorConfig: Updated factor configuration
            
        Raises:
            ValueError: If factor code cannot be loaded or factor type is invalid
        """
        # Handle code string - load factor class dynamically
        if isinstance(factor, str):
            logger.info(f"| 📝 Loading factor from code for update...")
            factor_cls = await self._load_factor_from_code(factor)
            # Use empty config if not provided
            if config is None:
                config = {}
        # Handle Factor instance
        elif isinstance(factor, Factor):
            factor_cls = type(factor)
            # Use instance's config if config not provided
            if config is None:
                config = {}
        # Handle Factor class
        else:
            factor_cls = factor
        
        factor_config = await self.factor_context_manager.update(
            factor_cls, factor_config_dict=config, new_version=new_version, description=description
        )
        self._registered_configs[factor_config.type] = factor_config
        logger.info(f"| ✅ Factor updated: {factor_config.type}")
        return factor_config
    
    async def copy(self, factor_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, new_config: Optional[Dict[str, Any]] = None) -> FactorConfig:
        """Copy an existing factor
        
        Args:
            factor_name: Name (type) of the factor to copy
            new_name: New name (type) for the copied factor. If None, uses original name.
            new_version: New version for the copied factor. If None, increments version.
            new_config: New configuration dict for the copied factor. If None, uses original config.
            
        Returns:
            FactorConfig: New factor configuration
        """
        factor_config = await self.factor_context_manager.copy(
            factor_name, new_name, new_version, new_config
        )
        self._registered_configs[factor_config.type] = factor_config
        return factor_config
    
    async def unregister(self, factor_name: str) -> bool:
        """Unregister a factor
        
        Args:
            factor_name: Name (type) of the factor to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.factor_context_manager.unregister(factor_name)
        if success and factor_name in self._registered_configs:
            del self._registered_configs[factor_name]
        return success
    
    async def restore(self, factor_name: str, version: str, auto_initialize: bool = True) -> Optional[FactorConfig]:
        """Restore a specific version of a factor from history
        
        Args:
            factor_name: Name (type) of the factor
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored factor
            
        Returns:
            FactorConfig of the restored version, or None if not found
        """
        factor_config = await self.factor_context_manager.restore(factor_name, version, auto_initialize)
        if factor_config:
            self._registered_configs[factor_config.type] = factor_config
        return factor_config
    
    async def __call__(self, factor_name: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Call a factor by name
        
        Args:
            factor_name: Factor name (type)
            df: Input DataFrame containing price data
            **kwargs: Additional keyword arguments passed to the factor's __call__
                      (e.g., periods=[10, 20, 50] to override default periods)
            
        Returns:
            pd.DataFrame: DataFrame containing the calculated factor values
        """
        factor = await self.get(factor_name)
        if factor is None:
            raise ValueError(f"Factor {factor_name} not found")
        return await factor(df, **kwargs)


# Global Factor manager instance
factor_manager = FactorManager()

