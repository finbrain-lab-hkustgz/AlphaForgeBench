import os
from typing import Union

def get_project_root() -> str:
    """Get the project root directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def assemble_project_path(path: str) -> str:
    """Assemble a path relative to the project root directory
    
    Args:
        path: Path string (relative or absolute)
        
    Returns:
        str: Absolute path string
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    else:
        return os.path.abspath(os.path.join(get_project_root(), path))