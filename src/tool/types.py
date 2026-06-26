from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field
from src.dynamic import dynamic_manager

class ToolExtra(BaseModel):
    """Extra data for a tool response"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    file_path: Optional[Union[str, List[str]]] = Field(default=None, description="The file path of the extra data")
    data: Optional[Dict[str, Any]] = Field(default=None, description="The data of the extra data")
    parsed_model: Optional[BaseModel] = Field(default=None, description="The parsed model of the extra data")

class ToolResponse(BaseModel):
    """Response for a tool call."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    success: bool = Field(description="Whether the tool call was successful")
    message: str = Field(description="The message from the tool call")
    extra: Optional[ToolExtra] = Field(default=None, description="The extra data from the tool call")

class Tool(BaseModel):
    """Base class for all tools that can be exposed through function calling."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="The metadata of the tool")
    
    async def __call__(self, **kwargs) -> ToolResponse:
        """Call the tool with the given arguments."""
        raise NotImplementedError("All tools must implement __call__")

class ToolConfig(BaseModel):
    """Tool configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="The metadata of the tool")
    version: str = Field(default="1.0.0", description="Version of the tool")
    
    cls: Optional[Type[Tool]] = Field(default=None, description="The class of the tool")
    config: Optional[Dict[str, Any]] = Field(default={}, description="The initialization configuration of the tool")
    instance: Optional[Tool] = Field(default=None, description="The instance of the tool")
    code: Optional[str] = Field(default=None, description="Source code for dynamically generated tool classes (used when cls cannot be imported from a module)")
    
    # Default representations
    function_calling: Optional[Dict[str, Any]] = Field(default=None, description="Default function calling representation")
    text: Optional[str] = Field(default=None, description="Default text representation")
    args_schema: Optional[Type[BaseModel]] = Field(default=None, description="Default args schema (BaseModel type)")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        
        result = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "version": self.version,
            
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
            
            "function_calling": self.function_calling,
            "text": self.text,
            "args_schema": dynamic_manager.serialize_args_schema(self.args_schema) if self.args_schema else None,
        }
        
        return result
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'ToolConfig':
        """Validate the model from a dictionary."""
        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata")
        version = data.get("version")
        
        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code, 
                        class_name=class_name,
                        base_class=Tool,
                        context="tool"
                    )
                except Exception as e:
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None
            
        config = data.get("config")
        instance = data.get("instance", None)

        function_calling = data.get("function_calling")
        text = data.get("text")
        args_schema = dynamic_manager.deserialize_args_schema(data.get("args_schema"))
        
        return cls(name=name, 
            description=description,
            metadata=metadata,
            version=version,
            cls=cls_, 
            config=config, 
            instance=instance, 
            function_calling=function_calling, 
            text=text, 
            args_schema=args_schema
        )

__all__ = [
    "Tool",
    "ToolResponse",
    "ToolConfig",
]

