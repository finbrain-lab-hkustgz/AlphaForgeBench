"""Types for dynamic module management."""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DynamicModuleInfo:
    """Information about a dynamically loaded module.
    
    Attributes:
        module_name: The virtual module name (e.g., "_dynamic_module_1")
        code: The source code that was executed
        loaded_classes: Dictionary of class names to class objects that were loaded
        loaded_functions: Dictionary of function names to function objects that were loaded
    """
    module_name: str
    code: str
    loaded_classes: Dict[str, type]
    loaded_functions: Dict[str, Any]

