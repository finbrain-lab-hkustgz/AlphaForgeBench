"""Version Manager

Unified version management system for factors, strategies, and tools.
Supports version tracking, evolution, and history management.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import json
import os

from src.logger import logger
from src.config import config
from src.utils import assemble_project_path
from src.utils.file_utils import file_lock

T = TypeVar('T', bound=BaseModel)


class VersionStatus(str, Enum):
    """Version status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VersionInfo(BaseModel):
    """Version information"""
    version: str = Field(description="Version string (e.g., '1.0.0', '2.1.3')")
    status: VersionStatus = Field(default=VersionStatus.ACTIVE, description="Version status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    description: Optional[str] = Field(default=None, description="Version description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Version metadata")


class ComponentVersionHistory(BaseModel):
    """Version history for a component (only version records, no configs)"""
    name: str = Field(description="Name of the component")
    component_type: str = Field(description="Type of component (factor, strategy, tool)")
    current_version: str = Field(description="Current active version")
    versions: Dict[str, VersionInfo] = Field(default_factory=dict, description="Version history records")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Component metadata")
    
    def add_version(self, version: str, description: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> VersionInfo:
        """Add a new version record
        
        Args:
            version: Version string
            description: Version description
            metadata: Version metadata
            
        Returns:
            VersionInfo: Created version info
        """
        if version in self.versions:
            logger.warning(f"| ⚠️ Version {version} already exists for {self.name}, updating...")
            version_info = self.versions[version]
            version_info.updated_at = datetime.now()
            if description:
                version_info.description = description
            if metadata:
                version_info.metadata.update(metadata)
        else:
            version_info = VersionInfo(
                version=version,
                description=description,
                metadata=metadata or {}
            )
            self.versions[version] = version_info
        
        self.current_version = version
        
        logger.debug(f"| ✅ Added version record {version} for {self.name}")
        return version_info
    
    def list_versions(self) -> List[str]:
        """List all available versions
        
        Returns:
            List of version strings
        """
        return list(self.versions.keys())
    
    def deprecate_version(self, version: str):
        """Deprecate a version
        
        Args:
            version: Version string to deprecate
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found for {self.name}")
        
        if version == self.current_version:
            raise ValueError(f"Cannot deprecate current version {version}")
        
        self.versions[version].status = VersionStatus.DEPRECATED
        logger.info(f"| 📝 Deprecated version {version} for {self.name}")
    
    def archive_version(self, version: str):
        """Archive a version
        
        Args:
            version: Version string to archive
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found for {self.name}")
        
        self.versions[version].status = VersionStatus.ARCHIVED
        logger.info(f"| 📦 Archived version {version} for {self.name}")


class VersionManager(BaseModel):
    """Unified version manager for all components - only manages version records"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the version histories")
    save_path: str = Field(default=None, description="The path to save version histories")
    
    def __init__(self, base_dir: Optional[str] = None, save_path: Optional[str] = None, **kwargs):
        """Initialize version manager"""
        super().__init__(**kwargs)
        
        # Storage: component_type -> name -> ComponentVersionHistory
        self._version_histories: Dict[str, Dict[str, ComponentVersionHistory]] = {
            "factor": {},
            "strategy": {},
            "tool": {}
        }
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "version"))
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "version.json")
        os.makedirs(self.base_dir, exist_ok=True)
        
        logger.info(f"| 📁 Version manager base directory: {self.base_dir} and save path: {self.save_path}")

    async def initialize(self):
        """Initialize version manager (for backward compatibility)"""
        logger.info(f"| 📁 Version manager initialized.")

    
    async def register_version(self, component_type: str, name: str, version: str,
                        description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ComponentVersionHistory:
        """Register a version record (only version info, no config)
        
        Args:
            component_type: Type of component (factor, strategy, tool)
            name: Component name
            version: Version string
            description: Version description
            metadata: Version metadata
            
        Returns:
            ComponentVersionHistory: Version history for the component
        """
        if component_type not in self._version_histories:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if name not in self._version_histories[component_type]:
            version_history = ComponentVersionHistory(
                name=name,
                component_type=component_type,
                current_version=version
            )
            self._version_histories[component_type][name] = version_history
        else:
            version_history = self._version_histories[component_type][name]
        
        version_history.add_version(version, description, metadata)
        
        # Save to JSON after registering version
        await self.save_to_json()
        
        return version_history
    
    async def list(self) -> Dict[str, Dict[str, List[str]]]:
        """List all versions for all components
        
        Returns:
            Dictionary mapping component_type -> component_name -> list of versions
        """
        result = {}
        for component_type, histories in self._version_histories.items():
            result[component_type] = {}
            for name, version_history in histories.items():
                result[component_type][name] = version_history.list_versions()
        return result
    
    async def get_version_history(self, component_type: str, name: str) -> Optional[ComponentVersionHistory]:
        """Get version history for a component
        
        Args:
            component_type: Type of component
            name: Component name
            
        Returns:
            ComponentVersionHistory or None if not found
        """
        if component_type not in self._version_histories:
            return None
        
        return self._version_histories[component_type].get(name)
    
    async def get_current_version(self, component_type: str, name: str) -> Optional[str]:
        """Get current version for a component
        
        Args:
            component_type: Type of component
            name: Component name
            
        Returns:
            Current version string or None if not found
        """
        version_history = await self.get_version_history(component_type, name)
        if version_history is None:
            return None
        return version_history.current_version
    
    async def generate_next_version(self, component_type: str, name: str, 
                                   version_type: str = "patch") -> str:
        """Generate next version number for a component
        
        Args:
            component_type: Type of component (factor, strategy, tool)
            name: Component name
            version_type: Type of version increment ("major", "minor", "patch")
            
        Returns:
            Next version string (e.g., "1.0.1", "1.1.0", "2.0.0")
        """
        current_version = await self.get_current_version(component_type, name)
        
        if current_version is None:
            # First version
            return "1.0.0"
        
        try:
            # Parse current version (e.g., "1.2.3")
            version_parts = current_version.split(".")
            if len(version_parts) >= 3:
                major = int(version_parts[0])
                minor = int(version_parts[1])
                patch = int(version_parts[2])
            elif len(version_parts) == 2:
                major = int(version_parts[0])
                minor = int(version_parts[1])
                patch = 0
            elif len(version_parts) == 1:
                major = int(version_parts[0])
                minor = 0
                patch = 0
            else:
                # Invalid version format, start fresh
                return "1.0.0"
            
            # Increment based on version_type
            if version_type == "major":
                major += 1
                minor = 0
                patch = 0
            elif version_type == "minor":
                minor += 1
                patch = 0
            else:  # patch (default)
                patch += 1
            
            return f"{major}.{minor}.{patch}"
            
        except (ValueError, IndexError):
            # If version parsing fails, start fresh
            logger.warning(f"| ⚠️ Failed to parse version {current_version} for {component_type}/{name}, starting fresh")
            return "1.0.0"
    
    async def get_version(self, 
                          component_type: str, 
                          name: str, 
                          provided_version: Optional[str] = None) -> str:
        """Get version from version_manager or generate new one
        
        Args:
            component_type: Type of component (factor, strategy, tool)
            name: Component name
            provided_version: Version provided by user (if any)
            
        Returns:
            Version string to use
        """
        if provided_version:
            # Use provided version
            return provided_version
        
        # Check if component already exists
        current_version = await self.get_current_version(component_type, name)
        if current_version is None:
            # New component, start with 1.0.0
            return "1.0.0"
        else:
            # Existing component, generate next patch version
            return await self.generate_next_version(component_type, name, "patch")
    
    async def deprecate_version(self, component_type: str, name: str, version: str):
        """Deprecate a version for a component
        
        Args:
            component_type: Type of component
            name: Component name
            version: Version string to deprecate
        """
        version_history = await self.get_version_history(component_type, name)
        if version_history is None:
            raise ValueError(f"Component {component_type}/{name} not found")
        
        version_history.deprecate_version(version)
        
        # Save to JSON after deprecating version
        await self.save_to_json()
    
    async def archive_version(self, component_type: str, name: str, version: str):
        """Archive a version for a component
        
        Args:
            component_type: Type of component
            name: Component name
            version: Version string to archive
        """
        version_history = await self.get_version_history(component_type, name)
        if version_history is None:
            raise ValueError(f"Component {component_type}/{name} not found")
        
        version_history.archive_version(version)
        
        # Save to JSON after archiving version
        await self.save_to_json()
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all version histories to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Serialize all version histories
            save_data = {
                "component_type": {},
                "metadata": {
                    "saved_at": datetime.now().isoformat()
                }
            }
            
            for component_type, histories in self._version_histories.items():
                save_data["component_type"][component_type] = {}
                for name, version_history in histories.items():
                    # Convert to dict
                    history_dict = version_history.model_dump(mode="json")
                    save_data["component_type"][component_type][name] = history_dict
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved version histories to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load version histories from JSON
        
        Args:
            file_path: File path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Version file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                # Clear existing histories
                for component_type in self._version_histories:
                    self._version_histories[component_type].clear()
                
                # Load histories
                component_types = load_data.get("component_type", {})
                for component_type, histories in component_types.items():
                    if component_type not in self._version_histories:
                        logger.warning(f"| ⚠️ Unknown component type: {component_type}")
                        continue
                    
                    for name, history_dict in histories.items():
                        try:
                            # Reconstruct ComponentVersionHistory
                            version_history = ComponentVersionHistory(**history_dict)
                            self._version_histories[component_type][name] = version_history
                        except Exception as e:
                            logger.error(f"| ❌ Failed to load version history for {name}: {e}")
                            continue
                
                logger.info(f"| 📂 Loaded version histories from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load version data from {file_path}: {e}")
                return False

    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """Compare two version strings. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal.
        
        Args:
            v1: First version string (e.g., "1.0.0")
            v2: Second version string (e.g., "1.0.1")
            
        Returns:
            1 if v1 > v2, -1 if v1 < v2, 0 if equal
        """
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]
            
            # Pad with zeros to same length
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))
            
            for p1, p2 in zip(parts1, parts2):
                if p1 > p2:
                    return 1
                elif p1 < p2:
                    return -1
            return 0
        except:
            # Fallback: string comparison
            return 1 if v1 > v2 else (-1 if v1 < v2 else 0)


# Global version manager instance
version_manager = VersionManager()
