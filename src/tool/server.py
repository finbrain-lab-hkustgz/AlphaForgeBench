"""TCP Server

Server implementation for the Tool Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
import os
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.tool.context import ToolContextManager
from src.tool.types import Tool, ToolConfig, ToolResponse
from src.utils import assemble_project_path

class ToolManager(BaseModel):
    """Tool Manager for managing tool registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    contract_path: str = Field(default=None, description="The path to save the tool contract")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the Tool Manager."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, ToolConfig] = {}  # tool_name -> ToolConfig

        
    async def initialize(self, tool_names: Optional[List[str]] = None):
        """Initialize tools by names using tool context manager with concurrent support.
        
        Args:
            tool_names: List of tool names to initialize. If None, initialize all registered tools.
        """
        
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "tool"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "tool.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Tool Manager base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")
        
        # Initialize tool context manager
        self.tool_context_manager = ToolContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
            model_name="openrouter/gemini-3-flash-preview",
            embedding_model_name="openrouter/text-embedding-3-large",
        )
        await self.tool_context_manager.initialize(tool_names=tool_names)
        
        logger.info("| ✅ Tools initialization completed")
        
    async def get_contract(self) -> str:
        """Get the contract for all tools"""
        return await self.tool_context_manager.load_contract()
    
    async def register(self, 
                       tool: Union[Tool, Type[Tool]],
                       config: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> ToolConfig:
        """Register a tool class or instance asynchronously.
        
        Args:
            tool: Tool class or instance to register
            config: Configuration dict for tool initialization (required when tool is a class)
            override: Whether to override existing registration
            version: Optional version string
            
        Returns:
            ToolConfig: Tool configuration
        """
        tool_config = await self.tool_context_manager.register(
            tool, 
            tool_config_dict=config, 
            override=override,
            version=version
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def list(self) -> List[str]:
        """List all registered tools
        
        Args:
            include_disabled: Whether to include disabled tools
            
        Returns:
            List[str]: List of tool names
        """
        return await self.tool_context_manager.list()
    
    
    async def get(self, tool_name: str) -> Tool:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool: Tool instance or None if not found
        """
        tool = await self.tool_context_manager.get(tool_name)
        return tool
    
    async def get_info(self, tool_name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolConfig: Tool configuration or None if not found
        """
        return await self.tool_context_manager.get_info(tool_name)
    
    async def cleanup(self):
        """Cleanup all tools"""
        await self.tool_context_manager.cleanup()
        self._registered_configs.clear()
    
    async def update(self, 
                     tool_name: str, tool: Union[Tool, Type[Tool]], 
                     config: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> ToolConfig:
        """Update an existing tool with new configuration and create a new version
        
        Args:
            tool_name: Name of the tool to update
            tool: New tool class or instance with updated implementation
            config: Configuration dict for tool initialization (required when tool is a class)
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        tool_config = await self.tool_context_manager.update(
            tool_name, tool, tool_config_dict=config, new_version=new_version, description=description
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def copy(self, tool_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> ToolConfig:
        """Copy an existing tool
        
        Args:
            tool_name: Name of the tool to copy
            new_name: New name for the copied tool. If None, uses original name.
            new_version: New version for the copied tool. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            ToolConfig: New tool configuration
        """
        tool_config = await self.tool_context_manager.copy(
            tool_name, new_name, new_version, **override_config
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def unregister(self, tool_name: str) -> bool:
        """Unregister a tool
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.tool_context_manager.unregister(tool_name)
        if success and tool_name in self._registered_configs:
            del self._registered_configs[tool_name]
        return success
    
    async def restore(self, tool_name: str, version: str, auto_initialize: bool = True) -> Optional[ToolConfig]:
        """Restore a specific version of a tool from history
        
        Args:
            tool_name: Name of the tool
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored tool
            
        Returns:
            ToolConfig of the restored version, or None if not found
        """
        tool_config = await self.tool_context_manager.restore(tool_name, version, auto_initialize)
        if tool_config:
            self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def __call__(self, name: str, input: Dict[str, Any]) -> ToolResponse:
        """Call a tool by name
        
        Args:
            name: Tool name
            input: Input for the tool
            
        Returns:
            ToolResponse: Tool result
        """
        return await self.tool_context_manager(name, input)


# Global Tool manager instance
tool_manager = ToolManager()
